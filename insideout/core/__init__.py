"""Core models, emotions, and prompts."""

from insideout.core.models import (
    DialogueTurn,
    Dialogue,
    ERCAgentOutput,
    ERCAggregateOutput,
    ERGCandidateResponse,
    ERGAggregateOutput,
    ExperimentConfig,
    ERCResult,
    ERGResult
)

from insideout.core.emotions import (
    EmotionType,
    EKMAN_EMOTIONS,
    get_labels_for_set,
    map_label_32_to_18
)

from insideout.core.prompts import (
    # ERC formatters
    format_erc_emotional_prompt,
    format_erc_aggregate_prompt,
    format_erc_baseline_prompt,
    # ERG formatters
    format_erg_zeroshot_prompt,
    format_erg_emotional_prompt,
    format_erg_aggregate_prompt,
    format_erg_baseline_prompt
)

__all__ = [
    # Models
    "DialogueTurn",
    "Dialogue",
    "ERCAgentOutput",
    "ERCAggregateOutput",
    "ERGCandidateResponse",
    "ERGAggregateOutput",
    "ExperimentConfig",
    "ERCResult",
    "ERGResult",
    # Emotions
    "EmotionType",
    "EKMAN_EMOTIONS",
    "get_labels_for_set",
    "map_label_32_to_18",
    # Prompts
    "format_erc_emotional_prompt",
    "format_erc_aggregate_prompt",
    "format_erc_baseline_prompt",
    "format_erg_zeroshot_prompt",
    "format_erg_emotional_prompt",
    "format_erg_aggregate_prompt",
    "format_erg_baseline_prompt"
]
