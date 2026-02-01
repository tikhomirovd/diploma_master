"""
Pydantic models for InsideOut framework.

All data structures used in ERC and ERG pipelines.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from insideout.core.emotions import EmotionType


class DialogueTurn(BaseModel):
    """A single turn in a dialogue."""
    
    speaker_idx: int = Field(
        ...,
        description="Speaker index (0 or 1, typically 1=seeker, 0=listener)"
    )
    utterance: str = Field(
        ...,
        description="The text content of this turn"
    )
    
    class Config:
        frozen = False


class Dialogue(BaseModel):
    """
    A complete dialogue with context and metadata.
    
    Represents a multi-turn conversation from EmpatheticDialogues dataset.
    """
    
    conv_id: str = Field(
        ...,
        description="Unique conversation ID"
    )
    turns: List[DialogueTurn] = Field(
        ...,
        description="Ordered list of dialogue turns"
    )
    emotion_label: str = Field(
        ...,
        description="Ground truth emotion label (for ERC evaluation)"
    )
    prompt: str = Field(
        ...,
        description="Context/situation prompt that initiated the dialogue"
    )
    
    @field_validator("turns")
    @classmethod
    def validate_turns_not_empty(cls, v):
        if not v:
            raise ValueError("Dialogue must have at least one turn")
        return v
    
    def get_dialogue_history_str(self, max_turns: Optional[int] = None) -> str:
        """
        Format dialogue history as a string.
        
        Args:
            max_turns: If specified, only include the last N turns
        
        Returns:
            Formatted dialogue string
        """
        turns_to_use = self.turns[-max_turns:] if max_turns else self.turns
        lines = []
        for turn in turns_to_use:
            speaker = f"Speaker {turn.speaker_idx}"
            lines.append(f"{speaker}: {turn.utterance}")
        return "\n".join(lines)
    
    class Config:
        frozen = False


# ============================================================================
# ERC (Emotion Recognition in Conversation) Models
# ============================================================================

class ERCAgentOutput(BaseModel):
    """Output from a single emotional agent for ERC task."""
    
    agent_emotion: EmotionType = Field(
        ...,
        description="The emotion this agent embodies (anger, disgust, fear, happiness, sadness)"
    )
    predicted_label: str = Field(
        ...,
        description="Predicted emotion label for the speaker"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in [0, 1]"
    )
    rationale: str = Field(
        ...,
        description="Reasoning for the prediction from this emotion's perspective"
    )
    
    class Config:
        frozen = False


class ERCAggregateOutput(BaseModel):
    """Aggregated output from all emotional agents for ERC task."""
    
    final_label: str = Field(
        ...,
        description="Final predicted emotion label after aggregation"
    )
    final_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregated confidence score"
    )
    final_rationale: str = Field(
        ...,
        description="Synthesized rationale combining all agents' reasoning"
    )
    agent_outputs: List[ERCAgentOutput] = Field(
        ...,
        description="Individual outputs from all emotional agents"
    )
    
    @field_validator("agent_outputs")
    @classmethod
    def validate_agent_outputs_not_empty(cls, v):
        if not v:
            raise ValueError("agent_outputs must contain at least one output")
        return v
    
    class Config:
        frozen = False


# ============================================================================
# ERG (Empathetic Response Generation) Models
# ============================================================================

class ERGCandidateResponse(BaseModel):
    """A candidate empathetic response from a single emotional agent."""
    
    agent_emotion: EmotionType = Field(
        ...,
        description="The emotion this agent embodies"
    )
    response: str = Field(
        ...,
        description="Generated empathetic response text"
    )
    reasoning: str = Field(
        ...,
        description="Why this response is appropriate from this emotion's perspective"
    )
    
    class Config:
        frozen = False


class ERGAggregateOutput(BaseModel):
    """Aggregated output from all emotional agents for ERG task."""
    
    predicted_emotion: str = Field(
        ...,
        description="Predicted emotion of the speaker (from zero-shot step)"
    )
    candidates: List[ERGCandidateResponse] = Field(
        ...,
        description="Candidate responses from all emotional agents"
    )
    final_response: str = Field(
        ...,
        description="Selected best empathetic response"
    )
    selection_reasoning: str = Field(
        ...,
        description="Why this response was selected as the best"
    )
    
    @field_validator("candidates")
    @classmethod
    def validate_candidates_not_empty(cls, v):
        if not v:
            raise ValueError("candidates must contain at least one response")
        return v
    
    class Config:
        frozen = False


# ============================================================================
# Experiment and Results Models
# ============================================================================

class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run."""
    
    model_config = {"protected_namespaces": ()}
    
    experiment_name: str = Field(
        ...,
        description="Name/identifier for this experiment"
    )
    model_provider: str = Field(
        ...,
        description="LLM provider (gigachat, openai, etc.)"
    )
    model_name: str = Field(
        ...,
        description="Specific model name/version"
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature (paper uses 0.0)"
    )
    task_type: str = Field(
        ...,
        description="Task type: 'erc' or 'erg'"
    )
    method: str = Field(
        ...,
        description="Method: 'baseline' or 'insideout'"
    )
    label_set: str = Field(
        ...,
        description="Label set for ERC: '32' or '18'"
    )
    split: str = Field(
        default="test",
        description="Dataset split: 'train', 'val', or 'test'"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit number of samples (for testing)"
    )


class ERCResult(BaseModel):
    """Result for a single ERC prediction."""
    
    conv_id: str
    ground_truth: str
    predicted: str
    confidence: float
    rationale: Optional[str] = None
    
    @property
    def is_correct(self) -> bool:
        """Check if prediction matches ground truth."""
        return self.ground_truth == self.predicted
    
    class Config:
        frozen = False


class ERGResult(BaseModel):
    """Result for a single ERG generation."""
    
    conv_id: str
    reference_response: str
    generated_response: str
    predicted_emotion: Optional[str] = None
    
    class Config:
        frozen = False
