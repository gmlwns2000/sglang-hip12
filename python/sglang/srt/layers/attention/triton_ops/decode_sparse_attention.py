from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import os
import torch
from torch import Tensor
import triton
import triton.language as tl

from hip.models.hip_attention.gen3.uvm_gpu_cache import load_tokens
from hip.models.hip_attention.gen3.attention_metadata import safe_stride
from hip.models.hip_attention.gen3.attention_extend_bsa import block_sparse_attention_cuda_step

if TYPE_CHECKING:
    from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionArgs

DEFAULT_EXTEND_BACKEND: tl.constexpr = 'streaming'
MAX_INT: tl.constexpr = 2_147_483_647


@triton.jit
def _fwd_kernel_stage1(
    Q, stride_q_bsz, stride_q_tdst, stride_q_head, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
    V, stride_v_bsz, stride_v_tsrc, stride_v_head, stride_v_hid,
    B_Seqlen, stride_pos_bsz, stride_pos_tdst,

    INDICES,
    stride_indices_b, stride_indices_bdst, stride_indices_bk,

    KS_START_END,
    stride_ks_start_end_b, stride_ks_start_end_bdst, stride_ks_start_end_g,

    ATTN_LOGITS,
    stride_attn_logits_bsz, stride_attn_logits_head, stride_attn_logits_kv_split, stride_attn_logits_hid,

    CONTEXT,
    stride_context_bsz,
    stride_context_tdst,
    stride_context_head,
    stride_context_hid,

    q_head_num: tl.constexpr,
    BK: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    kv_group_num: tl.constexpr,

    sliding_window_size: tl.constexpr,
    sink_token_size: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,

    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    model_context_length,

    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,

    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,

    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,

    TDST_NEXT_POWER_OF_2,

    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    Lk: tl.constexpr,  # hidden dim of key
    Lv: tl.constexpr,  # hidden dim of value

    # autotuning parameters
    BLOCK_BK: tl.constexpr,  # = BLOCK_N / BLOCK_SIZE_K
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch * stride_pos_bsz + 0 * stride_pos_tdst)
    # cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_q = (
        cur_batch * stride_q_bsz
        + 0 * stride_q_tdst
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :] * stride_q_hid
    )
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
    if q.dtype == tl.float8e5:
        q = q.to(tl.float16)

    if USING_EXTEND and NEED_APPLY_ROPE:
        rope_tdst = cur_batch_seq_len - 1

        queries_rot = tl.load(
            Q
            + cur_batch * stride_q_bsz
            + 0 * stride_q_tdst
            + cur_head[:, None] * stride_q_head
            + ((offs_d[None, :] + Lk // 2) % Lk) * stride_q_hid,
        )
        if queries_rot.dtype == tl.float8e5:
            queries_rot = queries_rot.to(tl.float16)

        cos_new = tl.load(
            COS
            + rope_tdst.to(tl.int64) * stride_cos_t
            + (offs_d[None, :] % (Lk // 2)) * stride_cos_hid,
        ).to(q.dtype)
        sin_new = tl.load(
            SIN
            + rope_tdst.to(tl.int64) * stride_sin_t +
            + (offs_d[None, :] % (Lk // 2)) * stride_sin_hid,
        ).to(q.dtype)

        queries_rot = queries_rot * (((cur_head[:, None] + Lk // 2) < Lk) * (-2) + 1).to(q.dtype)
        q = (q * cos_new + queries_rot * sin_new).to(q.dtype)

    # Start and end indices to the `indices` tensor
    range_start = tl.load(
        KS_START_END
        + cur_batch * stride_ks_start_end_b
        + 0 * stride_ks_start_end_bdst
        + 0 * stride_ks_start_end_g
    )
    range_end = tl.load(
        KS_START_END
        + cur_batch * stride_ks_start_end_b
        + 0 * stride_ks_start_end_bdst
        + 1 * stride_ks_start_end_g
    )

    kv_blocks_per_split = tl.cdiv(BK, NUM_KV_SPLITS)
    split_kv_block_start = kv_blocks_per_split * split_kv_id
    split_kv_block_end = tl.minimum(split_kv_block_start + kv_blocks_per_split, BK)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")  # m_i
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)  # l_i
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_block_end > split_kv_block_start:
        for i_bk in range(split_kv_block_start, split_kv_block_end, BLOCK_BK):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)
            mask_bk = (range_start <= idx_bk) & (idx_bk < tl.minimum(range_start + BK, range_end))

            if (range_start <= i_bk + BLOCK_BK) & (i_bk < range_end):
                idx_tsrc_start = tl.load(
                    INDICES
                    + cur_batch * stride_indices_b
                    + 0 * stride_indices_bdst
                    + idx_bk * stride_indices_bk,
                    mask=mask_bk,
                )
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full((1, BLOCK_SIZE_K), 1, dtype=tl.int1)
                mask_tsrc_from_bk = tl.reshape(mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc = ((MAX_TSRC * 0) <= idx_tsrc) & (idx_tsrc < (MAX_TSRC * 1)) & mask_tsrc_from_bk
                idx_tsrc = idx_tsrc % MAX_TSRC
                mask_tsrc = (sink_token_size <= idx_tsrc) & (idx_tsrc < cur_batch_seq_len) & mask_tsrc

                keys = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,

                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,

                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,

                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,

                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,

                    cur_batch,
                    idx_tsrc[None, :],
                    cur_kv_head,
                    offs_d[:, None],
                    mask_tsrc[None, :],

                    BLOCK_SIZE_K,
                    Lk,
                )

                if USING_EXTEND and NEED_APPLY_ROPE:
                    keys_rot = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,

                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,

                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,

                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,

                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,

                        cur_batch,
                        idx_tsrc[None, :],
                        cur_kv_head,
                        ((offs_d[:, None] + Lk // 2) % Lk)[:, None],
                        mask_tsrc[None, :],

                        BLOCK_SIZE_K,
                        Lk,
                    )
                else:
                    keys_rot = None

                values = load_tokens(
                    V,
                    stride_v_bsz,
                    stride_v_tsrc,
                    stride_v_head,
                    stride_v_hid,

                    USING_PAGES,
                    PAGE_SIZE,
                    V_CACHE,
                    stride_v_cache_page,
                    stride_v_cache_offset,
                    stride_v_cache_kv_head,
                    stride_v_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,

                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    True,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,

                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,

                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,

                    cur_batch,
                    idx_tsrc[:, None],
                    cur_kv_head,
                    offs_dv[None, :],
                    mask_tsrc[:, None],

                    BLOCK_SIZE_K,
                    Lv,
                )

                acc, e_sum, e_max = block_sparse_attention_cuda_step(
                    q,
                    keys,
                    keys_rot,
                    values,

                    idx_tsrc, mask_tsrc,
                    0, 1,

                    acc, e_sum, e_max,

                    sliding_window_size,
                    sink_token_size,
                    (range_end - range_start) * BLOCK_SIZE_K,
                    True,
                    False,
                    LOGIT_SOFTCAP,

                    USING_EXTEND,
                    NEED_APPLY_ROPE,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    model_context_length,

                    idx_bk + sink_token_size // BLOCK_SIZE_K,
                    cur_batch_seq_len,
                    offs_d,
                    IS_CAUSAL,
                    Lk,
                    BLOCK_SIZE_Q,
                    BLOCK_BK * BLOCK_SIZE_K,
                    BLOCK_SIZE_K,

                    EXTEND_BACKEND=EXTEND_BACKEND,
                )
            else:
                pass

    # TODO: process sink tokens

    # TODO: process sliding window

    # Store results
    offs_mid_o = (
        cur_batch * stride_attn_logits_bsz
        + cur_head[:, None] * stride_attn_logits_head
        + split_kv_id * stride_attn_logits_kv_split
        + offs_dv[None, :] * stride_attn_logits_hid
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o,
        acc / e_sum[:, None],
        mask=(mask_h[:, None]) & (mask_dv[None, :]),
    )

    offs_mid_o_1 = (
        cur_batch * stride_attn_logits_bsz
        + cur_head * stride_attn_logits_head
        + split_kv_id * stride_attn_logits_kv_split
        + Lv * stride_attn_logits_hid
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o_1,
        e_max + tl.log(e_sum),
        mask=mask_h,
    )


def decode_block_sparse_attention_stage1(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks_start_end: Tensor,
    context: Tensor,
    args: HiPAttentionArgs,
    head_num: int, BK: int, MAX_TDST: int, MAX_TSRC: int, kv_group_num: int,
    model_context_length: int,
    HID: int, BLOCK_BK: int, extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
):
    batch = q.shape[0]
    BLOCK_H = 16
    NUM_KV_SPLITS = 8  # TODO: apply from server args
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    BLOCK_DMODEL = triton.next_power_of_2(HID)
    BLOCK_DV = triton.next_power_of_2(HID)

    _fwd_kernel_stage1[grid](
        q, *safe_stride(q, 4),
        k, *safe_stride(k, 4),
        v, *safe_stride(v, 4),
        seq_lens, *safe_stride(seq_lens, 2),

        indices, *safe_stride(indices, 3),

        ks_start_end, *safe_stride(ks_start_end, 3),

        # TODO: add ATTN_LOGITS

        context, *safe_stride(context, 4),

        head_num, BK, MAX_TDST, MAX_TSRC, kv_group_num,

        args.sliding_window_size,
        args.sink_token_size,
        args.logit_softcap,

        *args.args_extend(),
        model_context_length,
        *args.args_paged_kv_cache(),
        *args.args_offload_cache(is_masking=False),

        access_counter, *safe_stride(access_counter, 3),
        cache_miss_counter, *safe_stride(cache_miss_counter, 3),

        triton.next_power_of_2(MAX_TDST),

        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        Lk=HID, Lv=HID,

        BLOCK_BK=BLOCK_BK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_H=BLOCK_H,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        EXTEND_BACKEND=extend_backend,
    )


def decode_block_sparse_attention_stage2():
    pass


def decode_block_sparse_attention_impl(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks_start_end: Tensor,
    context: Tensor,
    args: HiPAttentionArgs,
    HEAD: int, BK: int, MAX_TDST: int, MAX_TSRC: int, KV_HEAD_REPEAT: int,
    model_context_length: int,
    HID: int, BLOCK_BK: int, extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
):
    """
    FlashDecode block sparse attention.
    :param q: (BSZ, TDST, HEAD, HID)
    :param seq_lens: (BSZ, TDST)
    :param indices: (BSZ, TDST, BK)
    :param ks_start_end: (BSZ, BSRC, 2)
    :param context: (BSZ, TDST, HEAD, HID)
    """
    decode_block_sparse_attention_stage1(
        q, k, v,
        seq_lens=seq_lens,
        indices=indices,
        ks_start_end=ks_start_end,
        context=context,
        args=args,
        head_num=HEAD, BK=BK,
        MAX_TDST=MAX_TDST, MAX_TSRC=MAX_TSRC,
        kv_group_num=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID, BLOCK_BK=BLOCK_BK,
        extend_backend=extend_backend,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
    )
    decode_block_sparse_attention_stage2()


def decode_block_sparse_attention(
    q: Tensor,  # [1, 1 (TDST), 32 (Q_HEAD), 128]
    k: Optional[Tensor],  # None
    v: Optional[Tensor],  # None
    seq_lens: Tensor,  # [1, 1 (TDST)], tensor([34089])

    indices: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 512]
    ks: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST)]
    ks_count: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 1]
    ks_start_end: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 2]

    args: HiPAttentionArgs,
    # args.block_table: [1 (BSZ), 196612]
    # args.cache_seq_lens: [1 (BSZ)], tensor([34089])
    # args.k_cache: [109527 (NUM_PAGE), 1 (PAGE_SIZE), 8 (KV_HEAD), 128 (Lk)]
    # args.v_cache: [109527 (NUM_PAGE), 1 (PAGE_SIZE), 8 (KV_HEAD), 128 (Lv)]
    # args.position_ids: [1, 1 (TDST)]
    # args.rope_cos: [196608, 128 (Lk)]
    # args.rope_sin: [196608, 128 (Lk)]

    access_counter: Tensor,  # [1, 8, 109527]
    cache_miss_counter: Tensor,  # [1, 8, 109527]

    EXTEND_BACKEND: str = DEFAULT_EXTEND_BACKEND,  # 'streaming'
    model_context_length: int = 131072,  # 131072
    extend_context_length: int = 131072,  # 196608
):
    BSZ, TDST, HEAD, HID = q.shape

    assert TDST == 1, "TDST must be 1 for flashdecode"

    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        BSRC = triton.cdiv(TSRC, args.block_size_k)
        MAX_TSRC = TSRC
        MAX_BSRC = BSRC
    else:
        if args.k_cache is not None:
            NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
        else:
            KV_HEAD = args.offload_cache.k_uvm.bank_cpu.shape[-2]
        TSRC = None
        BSRC = None
        MAX_TSRC = extend_context_length
        MAX_BSRC = triton.cdiv(MAX_TSRC, args.block_size_k)
    N = BSZ * HEAD
    BDST = triton.cdiv(TDST, args.block_size_q)
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD

    B = N
    BK = indices.shape[-1]

    context = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    max_block_size = int(os.getenv('SA_BLOCK_SIZE', '32'))
    BLOCK_BK = max_block_size // args.block_size_k
    BLOCK_BK = max(1, min(max_block_size, BLOCK_BK))
    if 'SA_BLOCK_BK' in os.environ:
        BLOCK_BK = int(os.environ['SA_BLOCK_BK'])

    assert BLOCK_BK > 0, BLOCK_BK

    if args.rope_cos is not None:
        assert len(args.rope_cos.stride()) == 2
        assert len(args.rope_sin.stride()) == 2

    assert context.ndim == 4
    if ks_start_end is not None:
        assert ks_start_end.ndim == 3
    if indices is not None:
        assert indices.ndim == 3
    assert q.ndim == 4
    if k is not None:
        assert k.ndim == 4
        assert v.ndim == 4
    elif args.using_paged_cache:
        if args.k_cache is not None:
            assert args.k_cache.ndim == 4
            assert args.v_cache.ndim == 4
        else:
            assert args.offload_cache.k_uvm.bank_cpu.ndim == 3
            assert args.offload_cache.v_uvm.bank_cpu.ndim == 3
    else:
        raise Exception()
    assert seq_lens.ndim == 2

    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    decode_block_sparse_attention_impl(
        q, k, v,
        seq_lens=seq_lens,
        indices=indices,
        ks_start_end=ks_start_end,
        context=context,
        args=args,
        HEAD=HEAD, BK=BK,
        MAX_TDST=TDST, MAX_TSRC=MAX_TSRC,
        KV_HEAD_REPEAT=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID, BLOCK_BK=BLOCK_BK,
        extend_backend=EXTEND_BACKEND,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
    )

    torch.set_default_device(pre_device)

    return context
