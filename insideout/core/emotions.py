"""
Emotion types and mappings for InsideOut framework.

Based on Ekman's basic emotions used in the paper.
"""

from enum import Enum
from typing import List


class EmotionType(str, Enum):
    """Ekman's 5 basic emotions used for emotional agents."""
    
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"


# All Ekman emotions for iteration
EKMAN_EMOTIONS: List[EmotionType] = [
    EmotionType.ANGER,
    EmotionType.DISGUST,
    EmotionType.FEAR,
    EmotionType.HAPPINESS,
    EmotionType.SADNESS,
]


# EmpatheticDialogues 32 emotion labels
EMPATHETIC_DIALOGUES_32_LABELS = [
    "afraid",
    "angry",
    "annoyed",
    "anticipating",
    "anxious",
    "apprehensive",
    "ashamed",
    "caring",
    "confident",
    "content",
    "devastated",
    "disappointed",
    "disgusted",
    "embarrassed",
    "excited",
    "faithful",
    "furious",
    "grateful",
    "guilty",
    "hopeful",
    "impressed",
    "jealous",
    "joyful",
    "lonely",
    "nostalgic",
    "prepared",
    "proud",
    "sad",
    "sentimental",
    "surprised",
    "terrified",
    "trusting",
]


# EmpatheticDialogues 18 emotion subset (as used in paper experiments)
EMPATHETIC_DIALOGUES_18_LABELS = [
    "afraid",
    "angry",
    "annoyed",
    "anticipating",
    "anxious",
    "caring",
    "confident",
    "content",
    "excited",
    "faithful",
    "grateful",
    "hopeful",
    "impressed",
    "joyful",
    "proud",
    "sad",
    "surprised",
    "trusting",
]


# Mapping from 32 to 18 labels (simplified grouping)
LABEL_32_TO_18_MAPPING = {
    "afraid": "afraid",
    "angry": "angry",
    "annoyed": "annoyed",
    "anticipating": "anticipating",
    "anxious": "anxious",
    "apprehensive": "afraid",  # map to afraid
    "ashamed": "sad",  # map to sad
    "caring": "caring",
    "confident": "confident",
    "content": "content",
    "devastated": "sad",  # map to sad
    "disappointed": "sad",  # map to sad
    "disgusted": "annoyed",  # map to annoyed
    "embarrassed": "anxious",  # map to anxious
    "excited": "excited",
    "faithful": "faithful",
    "furious": "angry",  # map to angry
    "grateful": "grateful",
    "guilty": "sad",  # map to sad
    "hopeful": "hopeful",
    "impressed": "impressed",
    "jealous": "annoyed",  # map to annoyed
    "joyful": "joyful",
    "lonely": "sad",  # map to sad
    "nostalgic": "content",  # map to content
    "prepared": "confident",  # map to confident
    "proud": "proud",
    "sad": "sad",
    "sentimental": "content",  # map to content
    "surprised": "surprised",
    "terrified": "afraid",  # map to afraid
    "trusting": "trusting",
}


def get_labels_for_set(label_set: str) -> List[str]:
    """
    Get the list of emotion labels for a given label set.
    
    Args:
        label_set: Either "32" or "18"
    
    Returns:
        List of emotion label strings
    
    Raises:
        ValueError: If label_set is not "32" or "18"
    """
    if label_set == "32":
        return EMPATHETIC_DIALOGUES_32_LABELS
    elif label_set == "18":
        return EMPATHETIC_DIALOGUES_18_LABELS
    else:
        raise ValueError(f"Invalid label_set: {label_set}. Must be '32' or '18'")


def map_label_32_to_18(label: str) -> str:
    """
    Map a 32-class label to its 18-class equivalent.
    
    Args:
        label: Emotion label from 32-class set
    
    Returns:
        Corresponding label in 18-class set
    
    Raises:
        ValueError: If label not found in mapping
    """
    if label not in LABEL_32_TO_18_MAPPING:
        raise ValueError(f"Unknown label: {label}")
    return LABEL_32_TO_18_MAPPING[label]
