import os
from typing import Optional

import triton
import triton.language as tl
from hip.models.hip_attention.gen3.attention_extend_bsa import block_sparse_attention_cuda_step
from hip.models.hip_attention.gen3.attention_metadata import safe_stride
from hip.models.hip_attention.gen3.uvm_gpu_cache import load_tokens
import torch
from torch import Tensor

DEFAULT_EXTEND_BACKEND: tl.constexpr = 'streaming'
MAX_INT: tl.constexpr = 2_147_483_647


@triton.jit
def block_sparse_attention_cuda(
    Q, stride_q_bsz, stride_q_tdst, stride_q_head, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
    V, stride_v_bsz, stride_v_tsrc, stride_v_head, stride_v_hid,
    POS, stride_pos_bsz, stride_pos_tdst,

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

    HEAD: tl.constexpr,
    BK: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    KV_HEAD_REPEAT: tl.constexpr,

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
    HID: tl.constexpr,

    # autotuning parameters
    BLOCK_BK: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
):
    G: tl.constexpr = 1

    pid_bsz = tl.program_id(2).to(tl.int64)
    pid_bdst = tl.program_id(1).to(tl.int64)
    pid_head = tl.program_id(0).to(tl.int64)

    idx_bsz = pid_bsz.to(tl.int64)
    idx_head = pid_head
    idx_n = idx_bsz * HEAD + idx_head
    idx_b = idx_n
    idx_g = 0

    idx_bdst = pid_bdst
    if BLOCK_SIZE_Q < 16:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, 16)
        mask_tdst = (idx_tdst < MAX_TDST) & (tl.arange(0, 16) < BLOCK_SIZE_Q)
    else:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, BLOCK_SIZE_Q)
        mask_tdst = idx_tdst < MAX_TDST
    if IS_CAUSAL:
        pos_tdst = tl.load(
            POS + \
            idx_bsz * stride_pos_bsz + \
            idx_tdst * stride_pos_tdst,
            mask=mask_tdst,
            other=0,
        )
    else:
        pos_tdst = tl.where(
            mask_tdst,
            tl.full((BLOCK_SIZE_Q,), value=MAX_TSRC, dtype=tl.int64),
            0
        )

    idx_hid = tl.arange(0, HID)

    if BLOCK_SIZE_Q < 16:
        acc = tl.zeros((16, HID), dtype=tl.float32)
        m_i = tl.full((16, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((16, 1), 1.0, dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_Q, HID), dtype=tl.float32)
        m_i = tl.full((BLOCK_SIZE_Q, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_Q, 1), 1.0, dtype=tl.float32)

    range_start = tl.load(
        KS_START_END + \
        idx_b * stride_ks_start_end_b + \
        idx_bdst * stride_ks_start_end_bdst + \
        idx_g * stride_ks_start_end_g
    )
    range_end = tl.load(
        KS_START_END + \
        idx_b * stride_ks_start_end_b + \
        idx_bdst * stride_ks_start_end_bdst + \
        (idx_g + 1) * stride_ks_start_end_g
    )
    if BK <= 0:
        range_start = 0
        range_end = 0

    queries = tl.load(
        Q + \
        idx_bsz * stride_q_bsz + \
        idx_tdst[:, None] * stride_q_tdst + \
        idx_head * stride_q_head + \
        idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0.0,
    )
    if queries.dtype == tl.float8e5:
        queries = queries.to(tl.float16)

    if USING_EXTEND and NEED_APPLY_ROPE:
        rope_tdst = pos_tdst - 1

        queries_rot = tl.load(
            Q + \
            idx_bsz * stride_q_bsz + \
            idx_tdst[:, None] * stride_q_tdst + \
            idx_head * stride_q_head + \
            ((idx_hid + HID // 2) % HID)[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        )
        if queries_rot.dtype == tl.float8e5:
            queries_rot = queries_rot.to(tl.float16)

        cos_new = tl.load(
            COS + \
            rope_tdst[:, None].to(tl.int64) * stride_cos_t + \
            (idx_hid % (HID // 2))[None, :] * stride_cos_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)
        sin_new = tl.load(
            SIN + \
            rope_tdst[:, None].to(tl.int64) * stride_sin_t + \
            (idx_hid % (HID // 2))[None, :] * stride_sin_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)

        queries_rot = queries_rot * (((idx_hid + HID // 2)[None, :] < HID) * (-2) + 1).to(queries_rot.dtype)

        queries = (queries * cos_new + queries_rot * sin_new).to(queries.dtype)

    if (BK > 0) and True:
        for i_bk in range(range_start, range_start + (BK * G), BLOCK_BK):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)
            mask_bk = (idx_bk < (range_start + BK * G)) & (idx_bk < range_end)

            if i_bk < range_end:
                idx_tsrc_start = tl.load(
                    INDICES + \
                    idx_b * stride_indices_b + \
                    idx_bdst * stride_indices_bdst + \
                    idx_bk * stride_indices_bk,
                    mask=mask_bk,
                )
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC * G + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full((1, BLOCK_SIZE_K), 1, dtype=tl.int1)
                mask_tsrc_from_bk = tl.reshape(mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc = (idx_tsrc < (MAX_TSRC * (idx_g + 1))) & (
                    idx_tsrc >= (MAX_TSRC * idx_g)) & mask_tsrc_from_bk
                idx_tsrc = idx_tsrc % MAX_TSRC
                mask_tsrc = mask_tsrc & (idx_tsrc < tl.max(pos_tdst)) & (idx_tsrc >= sink_token_size)

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

                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid[:, None],
                    mask_tsrc[None, :],

                    BLOCK_SIZE_K,
                    HID,
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

                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        ((idx_hid + HID // 2) % HID)[:, None],
                        mask_tsrc[None, :],

                        BLOCK_SIZE_K,
                        HID,
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

                    idx_bsz,
                    idx_tsrc[:, None],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid[None, :],
                    mask_tsrc[:, None],

                    BLOCK_SIZE_K,
                    HID,
                )

                acc, l_i, m_i = block_sparse_attention_cuda_step(
                    queries,
                    keys,
                    keys_rot,
                    values,

                    idx_tsrc, mask_tsrc,
                    idx_tdst, mask_tdst,

                    acc, l_i, m_i,

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
                    pos_tdst,
                    idx_hid,
                    IS_CAUSAL,
                    HID,
                    BLOCK_SIZE_Q,
                    BLOCK_BK * BLOCK_SIZE_K,
                    BLOCK_SIZE_K,

                    EXTEND_BACKEND=EXTEND_BACKEND,
                )
            else:
                pass

    if (sink_token_size > 0) and True:
        for i_tsrc in range(0, sink_token_size, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < tl.minimum(MAX_TSRC, sink_token_size)

            # idx_n = idx_b * G + idx_group
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

                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],

                BLOCK_SIZE_K,
                HID,
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

                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    ((idx_hid + HID // 2) % HID)[:, None],
                    mask_tsrc[None, :],

                    BLOCK_SIZE_K,
                    HID,
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

                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],

                BLOCK_SIZE_K,
                HID,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                keys_rot,
                values,

                idx_tsrc, mask_tsrc,
                idx_tdst, mask_tdst,

                acc, l_i, m_i,

                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                True,
                True,
                LOGIT_SOFTCAP,

                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                model_context_length,

                tl.arange(0, BLOCK_BK) + \
                i_tsrc // BLOCK_SIZE_K,
                pos_tdst,
                idx_hid,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,

                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    if (sliding_window_size > 0):
        CURR_TSRC = tl.max(pos_tdst)
        # CURR_TSRC = (idx_bdst + 1) * BLOCK_SIZE_Q + MAX_TSRC - MAX_TDST
        i_tsrc_range_start = tl.maximum(0, CURR_TSRC - sliding_window_size - BLOCK_SIZE_Q)
        TSRC_RANGE_STEP: tl.constexpr = BLOCK_BK * BLOCK_SIZE_K
        for i_tsrc in range(i_tsrc_range_start, CURR_TSRC, TSRC_RANGE_STEP):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < MAX_TSRC

            # idx_n = idx_b * G + idx_group
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

                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],

                BLOCK_SIZE_K,
                HID,
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

                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    ((idx_hid + HID // 2) % HID)[:, None],
                    mask_tsrc[None, :],

                    BLOCK_SIZE_K,
                    HID,
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

                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],

                BLOCK_SIZE_K,
                HID,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                keys_rot,
                values,

                idx_tsrc, mask_tsrc,
                idx_tdst, mask_tdst,

                acc, l_i, m_i,

                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                False,
                False,
                LOGIT_SOFTCAP,

                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                model_context_length,

                tl.arange(0, BLOCK_BK) + \
                (i_tsrc - i_tsrc_range_start) // BLOCK_SIZE_K + \
                (tl.max(pos_tdst * mask_tdst) - tl.sum(
                    mask_tdst.to(tl.int32)) - sliding_window_size) // BLOCK_SIZE_K,
                pos_tdst,
                idx_hid,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,

                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    offs_mid_o = (
        idx_bsz * stride_attn_logits_bsz +
        idx_head * stride_attn_logits_head +
        0 * stride_attn_logits_kv_split +
        idx_hid[None, :] * stride_attn_logits_hid +
        idx_tdst[:, None] * 100000
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o,
        acc / (tl.where(l_i == 0.0, 1e-20, l_i)),
        mask=mask_tdst[:, None],
    )

    offs_mid_o_1 = (
        idx_bsz * stride_attn_logits_bsz +
        idx_head * stride_attn_logits_head +
        0 * stride_attn_logits_kv_split +
        HID * stride_attn_logits_hid +
        idx_tdst[:, None] * 100000
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o_1,
        m_i + tl.math.log2(tl.where(l_i == 0.0, 1e-20, l_i)),
        mask=mask_tdst[:, None],
    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = (acc / (tl.where(l_i == 0.0, 1e-20, l_i)))

    tl.store(
        CONTEXT + \
        idx_bsz * stride_context_bsz + \
        idx_tdst[:, None] * stride_context_tdst + \
        idx_head * stride_context_head + \
        idx_hid[None, :] * stride_context_hid,
        mask=mask_tdst[:, None],
        value=acc.to(CONTEXT.type.element_ty),
    )


def block_sparse_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,

    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,

    args: "HiPAttentionArgs",

    access_counter: Tensor,
    cache_miss_counter: Tensor,

    EXTEND_BACKEND: str = DEFAULT_EXTEND_BACKEND,
    model_context_length: int = 131072,
    extend_context_length: int = 131072,
):
    BSZ, TDST, HEAD, HID = q.shape
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
    assert B == N
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

    attn_logits = torch.zeros(
        (BSZ, HEAD, 1, HID + 1),
        dtype=q.dtype, device=q.device
    )

    grid = (HEAD, BDST, BSZ)
    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    block_sparse_attention_cuda[grid](
        q, *safe_stride(q, 4),
        k, *safe_stride(k, 4),
        v, *safe_stride(v, 4),
        seq_lens, *safe_stride(seq_lens, 2),

        indices, *safe_stride(indices, 3),

        ks_start_end, *safe_stride(ks_start_end, 3),

        attn_logits, *safe_stride(attn_logits, 4),

        context, *safe_stride(context, 4),

        HEAD, BK, TDST, MAX_TSRC, KV_HEAD_REPEAT,

        args.sliding_window_size,
        args.sink_token_size,
        args.logit_softcap,

        *args.args_extend(),
        model_context_length,
        *args.args_paged_kv_cache(),
        *args.args_offload_cache(is_masking=False),

        access_counter, *safe_stride(access_counter, 3),
        cache_miss_counter, *safe_stride(cache_miss_counter, 3),

        triton.next_power_of_2(TDST),

        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        HID,
        BLOCK_BK=BLOCK_BK,
        EXTEND_BACKEND=EXTEND_BACKEND,
    )
    torch.set_default_device(pre_device)

    return context, attn_logits
