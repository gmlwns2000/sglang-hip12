from dataclasses import dataclass, field, InitVar

from hip.models.hip_attention.attention2_draft_sampling_extend import ScanStage


_DEFAULT_STAGES = [
    ScanStage(
        stage_block_size_q=64,
        stage_block_stride_q=4,
        stage_chunk_size=256,
        stage_k=None,
        stage_stride=1,
    ),
    ScanStage(
        stage_block_size_q=64,
        stage_block_stride_q=4,
        stage_chunk_size=32,
        stage_k=32768,
        stage_stride=1,
    ),
    ScanStage(
        stage_block_size_q=64,
        stage_block_stride_q=1,
        stage_chunk_size=8,
        stage_k=8192,
        stage_stride=1,
    ),
]


@dataclass
class HiPAttentionPerLayerConfig:
    second_stage_k: int = 2048
    sliding_window_size: int = 1024
    sink_token_size: int = 256
    sa_extend_backend: str = 'streaming'
    stages: list = field(default_factory=lambda: _DEFAULT_STAGES)

    parsed_json: InitVar[dict | None] = None

    def __post_init__(self, parsed_json: dict | None):
        super().__init__()
        if parsed_json is not None:
            if 'second_stage_k' in parsed_json:
                self.second_stage_k = parsed_json['second_stage_k']
            if 'sliding_window_size' in parsed_json:
                self.sliding_window_size = parsed_json['sliding_window_size']
            if 'sink_token_size' in parsed_json:
                self.sink_token_size = parsed_json['sink_token_size']
            if 'sa_extend_backend' in parsed_json:
                self.sa_extend_backend = parsed_json['sa_extend_backend']
            if 'stages' in parsed_json:
                self.stages = [
                    ScanStage(**stage)
                    for stage in parsed_json['stages']
                ]


@dataclass
class HiPAttentionConfig:
    apply_v_dot: bool = False
    dense_layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    prefill_always_dense: bool = False
    decode_always_dense: bool = False
    force_dense: bool = False
    prefill_dense_threshold: int = 8192
    layers: list[HiPAttentionPerLayerConfig] = field(default_factory=lambda: [
        HiPAttentionPerLayerConfig(parsed_json={'second_stage_k': 4096, 'sliding_window_size': 8192, 'sink_token_size': 8192}),
        HiPAttentionPerLayerConfig(),
    ])

    parsed_json: InitVar[dict | None] = None

    def __post_init__(self, parsed_json: dict | None):
        super().__init__()
        if parsed_json is not None:
            if 'apply_v_dot' in parsed_json:
                self.apply_v_dot = parsed_json['apply_v_dot']
            if 'dense_layers' in parsed_json:
                self.dense_layers = parsed_json['dense_layers']
            if 'prefill_always_dense' in parsed_json:
                self.prefill_always_dense = parsed_json['prefill_always_dense']
            if 'decode_always_dense' in parsed_json:
                self.decode_always_dense = parsed_json['decode_always_dense']
            if 'force_dense' in parsed_json:
                self.force_dense = parsed_json['force_dense']
            if 'prefill_dense_threshold' in parsed_json:
                self.prefill_dense_threshold = parsed_json['prefill_dense_threshold']
            if 'layers' in parsed_json:
                self.layers = [
                    HiPAttentionPerLayerConfig(parsed_json=layer)
                    for layer in parsed_json['layers']
                ]
