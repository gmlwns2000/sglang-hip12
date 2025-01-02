from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional
import logging

import torch

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.hip_model_runner import HiPModelRunner
    from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig

from hip.models.hip_attention.gen3.attention_extend import dual_stage_quadratic_hip_attention
from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionArgs
from hip.models.hip_attention.gen3.uvm_gpu_cache import HiPOffloadCache


logger = logging.getLogger(__name__)


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


class HiPRadixAttentionBackend(AttentionBackend):

    def __init__(self, model_runner: HiPModelRunner):
        super().__init__()

        self.hip_config: HiPAttentionConfig = model_runner.hip_attention_config

        self.max_context_len = model_runner.model_config.context_len

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int):
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor = None,
    ):
        pass

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor = None,
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        # logger.info(f'HiP attention is used in prompting (layer {layer.layer_id})!', stacklevel=0)

        is_offload_cache = isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)

        if is_offload_cache:
            assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v,
                        async_copy=True, push_to_gpu_cache=False
                    )
            k_cache = v_cache = None
            # offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None
        else:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None

        q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)

        # Output tensor
        o = torch.empty_like(q_reshaped)

        start_len = 0
        decoding_reqs = []
        decoding_reqs_poistions = []
        for idx_batch, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if seq_len == 0:  # Skip empty sequences
                decoding_reqs.append(idx_batch)
                decoding_reqs_poistions.append(start_len)
            else:
                if is_offload_cache:
                    assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
                    k_chunk, v_chunk = forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                        layer_id=layer.layer_id,
                        batch_id=idx_batch,
                        cache_k=k[start_len:start_len+seq_len].unsqueeze(0),
                        cache_v=v[start_len:start_len+seq_len].unsqueeze(0),
                    )

                    # print(k_chunk.shape)

                    # BUG: test padding...
                    # _k_chunk = torch.zeros((1, 196608, k_chunk.shape[2], k_chunk.shape[3]), dtype=k_chunk.dtype, device=k_chunk.device)
                    # _v_chunk = torch.zeros((1, 196608, k_chunk.shape[2], k_chunk.shape[3]), dtype=k_chunk.dtype, device=k_chunk.device)
                    # _k_chunk[:, :k_chunk.shape[1], :, :] = k_chunk
                    # _v_chunk[:, :v_chunk.shape[1], :, :] = v_chunk
                    # k_chunk = _k_chunk
                    # v_chunk = _v_chunk

                    # print(k_chunk.shape)

                    k_cache = v_cache = None
                    offload_cache = None
                    # if layer.layer_id == 31:
                else:
                    k_chunk = v_chunk = None

                # print(layer.layer_id, k[::1,0,0], v[::1,0,0])

                o_req, _ = self.forward_paged_hip(
                    query=q_reshaped[start_len:start_len+seq_len],
                    sm_scale=layer.scaling,
                    batch_size=1,

                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,

                    positions=forward_batch.positions[start_len:start_len+seq_len],
                    seq_lens=forward_batch.seq_lens[idx_batch:idx_batch+1],
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices[idx_batch:idx_batch+1],

                    layer=layer,
                    is_dense=layer.layer_id in self.hip_config.dense_layers,

                    k=k_chunk,
                    v=v_chunk,
                )
                # if layer.layer_id == 31:
                # print(layer.layer_id, o_req[::1,0,0])
                o[start_len:start_len+seq_len] = o_req
            start_len += seq_len
        assert len(decoding_reqs) == 0

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        # logger.info(f'HiP attention is used in decoding (layer {layer.layer_id})!', stacklevel=0)

        is_offload_cache = isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)

        if is_offload_cache:
            assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v,
                        async_copy=False, push_to_gpu_cache=True,
                    )
            k_cache = v_cache = None
            offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        else:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None

        metadata = None
        if forward_batch.hip_use_cached_mask:
            metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                layer.layer_id, q.shape[0], forward_batch.batch_size
            )

        o, metadata = self.forward_paged_hip(
            query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            sm_scale=layer.scaling,
            batch_size=forward_batch.batch_size,

            k_cache=k_cache,
            v_cache=v_cache,
            offload_cache=offload_cache,

            positions=forward_batch.positions,
            seq_lens=forward_batch.seq_lens,
            req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,

            layer=layer,
            cached_metadata=metadata,
            is_dense=layer.layer_id in self.hip_config.dense_layers,
        )

        forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
            layer_id=layer.layer_id,
            size=q.shape[0],
            batch_size=forward_batch.batch_size,
            metadata=metadata,
        )

        if is_offload_cache:
            offload_cache.handle_cache_miss(metadata)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_paged_hip(
        self,
        query: torch.Tensor,
        sm_scale: float,
        batch_size: int,

        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        offload_cache: Optional[HiPOffloadCache],

        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_tokens: torch.Tensor,
        req_pool_indices: torch.Tensor,

        layer: RadixAttention,

        cached_metadata=None,
        is_dense: bool = False,

        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, "HiPAttentionOutputMetadata"]:
        is_dense = layer.layer_id in self.hip_config.dense_layers

        if len(self.hip_config.layers) == 2:
            layer_config = self.hip_config.layers[0 if is_dense else 1]
        else:
            layer_config = self.hip_config.layers[layer.layer_id]

        N, num_heads, hidden_dims = query.shape
        dst_seq_len = N // batch_size

        query = query.view(batch_size, dst_seq_len, num_heads, hidden_dims)

        if k_cache is not None:
            N_PAGE, num_heads_kv, hidden_dims_kv = k_cache.shape
            assert v_cache.shape == k_cache.shape
            assert hidden_dims_kv == hidden_dims

            k_cache = k_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)
            v_cache = v_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)

        block_table = req_to_tokens.index_select(dim=0, index=req_pool_indices)

        BLOCK_TABLE_BSZ, MODEL_SEQ_LEN = block_table.shape
        assert batch_size == BLOCK_TABLE_BSZ

        # NOTE(heejun): the whole point to need to find gemma is large size of hidden size
        # FIXME: find better way to detect Gemma
        if k_cache is not None:
            hidden_size = k_cache.shape[-1]
        elif k is not None:
            hidden_size = k.shape[-1]
        elif offload_cache is not None:
            hidden_size = offload_cache.k_uvm.bank_cpu.shape[-1]
        else:
            raise Exception()
        is_gemma = hidden_size > 128

        args = HiPAttentionArgs(
            k_cache=k_cache.view(torch.uint8) if isinstance(k_cache, torch.Tensor) and k_cache.dtype == torch.float8_e5m2 else k_cache,
            v_cache=v_cache.view(torch.uint8) if isinstance(k_cache, torch.Tensor) and v_cache.dtype == torch.float8_e5m2 else v_cache,
            offload_cache=offload_cache,
            block_table=block_table,
            cache_seq_lens=seq_lens,
            position_ids=positions.view(batch_size, dst_seq_len),

            block_size_k=32 if is_gemma else 64,  # BLOCK_CHUNK

            sliding_window_size=layer_config.sliding_window_size,
            sink_token_size=layer_config.sink_token_size,

            using_extend=True,
            need_apply_rope=True,
            rope_cos=layer.rope_cos,
            rope_sin=layer.rope_sin,

            logit_softcap=layer.logit_cap if layer.logit_cap != 0.0 else None,

            second_stage_k=layer_config.second_stage_k,
            stages=layer_config.stages,
            model_context_length=layer.orig_context_len,
            extend_context_length=self.max_context_len,
            block_sparse_block_size_q=self.hip_config.block_sparse_block_size_q,
            scan_extend_backend=('relative' if self.hip_config.apply_v_dot
                                 else ('streaming' if is_dense else 'relative')),
            sa_extend_backend=layer_config.sa_extend_backend,
        )

        print("===>", layer.layer_id, query.shape, args.k_cache.shape, args.v_cache.shape,
              args.block_table.shape,
              args.cache_seq_lens.shape, args.position_ids.shape,
              query, args.k_cache, args.v_cache, args.block_table, args.cache_seq_lens, args.position_ids,
              args.block_size_k, args.sliding_window_size, args.sink_token_size,
              args.using_extend, args.need_apply_rope, args.rope_cos.shape, args.rope_sin.shape, args.logit_softcap,
              layer_config.second_stage_k, layer_config.stages, layer.orig_context_len,
              cached_metadata,  # stage_args['cached_metadata'],
              args.block_sparse_block_size_q,
              args.scan_extend_backend,
              args.sa_extend_backend, sep="\n")

        # print(isinstance(k, torch.Tensor), isinstance(v, torch.Tensor), args.offload_cache, isinstance(args.k_cache, torch.Tensor))

        context, metadata = dual_stage_quadratic_hip_attention(
            (query * sm_scale).to(query.dtype),
            k, v,
            args=args,
            cached_metadata=cached_metadata,
        )
        context = context.to(query.dtype)

        print('context', context.shape)
        print(context)

        if layer.layer_id in [0, 1, 31]:
            from sglang.utils import ForkedPdb
            ForkedPdb().set_trace()

        return context.view(N, num_heads, hidden_dims), metadata
