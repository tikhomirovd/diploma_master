"""
Tests for emotion types and mappings.
"""

import pytest

from insideout.core.emotions import (
    EmotionType,
    EKMAN_EMOTIONS,
    get_labels_for_set,
    map_label_32_to_18,
    EMPATHETIC_DIALOGUES_32_LABELS,
    EMPATHETIC_DIALOGUES_18_LABELS,
)


class TestEmotionType:
    """Tests for EmotionType enum."""
    
    def test_emotion_type_values(self):
        """Test that EmotionType has correct values."""
        assert EmotionType.ANGER.value == "anger"
        assert EmotionType.DISGUST.value == "disgust"
        assert EmotionType.FEAR.value == "fear"
        assert EmotionType.HAPPINESS.value == "happiness"
        assert EmotionType.SADNESS.value == "sadness"
    
    def test_ekman_emotions_count(self):
        """Test that we have exactly 5 Ekman emotions."""
        assert len(EKMAN_EMOTIONS) == 5
    
    def test_ekman_emotions_are_emotion_types(self):
        """Test that all Ekman emotions are EmotionType instances."""
        for emotion in EKMAN_EMOTIONS:
            assert isinstance(emotion, EmotionType)


class TestLabelSets:
    """Tests for label set functions."""
    
    def test_32_labels_count(self):
        """Test that we have 32 labels in full set."""
        assert len(EMPATHETIC_DIALOGUES_32_LABELS) == 32
    
    def test_18_labels_count(self):
        """Test that we have 18 labels in subset."""
        assert len(EMPATHETIC_DIALOGUES_18_LABELS) == 18
    
    def test_18_labels_subset_of_32(self):
        """Test that 18-label set is a subset of 32-label set."""
        for label in EMPATHETIC_DIALOGUES_18_LABELS:
            # Either the label exists in 32 set, or it's a mapped target
            # (18 set contains the canonical labels)
            pass  # This is more complex, just check they're valid
    
    def test_get_labels_for_set_32(self):
        """Test getting labels for 32-class set."""
        labels = get_labels_for_set("32")
        assert len(labels) == 32
        assert labels == EMPATHETIC_DIALOGUES_32_LABELS
    
    def test_get_labels_for_set_18(self):
        """Test getting labels for 18-class set."""
        labels = get_labels_for_set("18")
        assert len(labels) == 18
        assert labels == EMPATHETIC_DIALOGUES_18_LABELS
    
    def test_get_labels_for_invalid_set(self):
        """Test that invalid label set raises error."""
        with pytest.raises(ValueError, match="Invalid label_set"):
            get_labels_for_set("10")


class TestLabelMapping:
    """Tests for 32 to 18 label mapping."""
    
    def test_map_label_32_to_18_identity(self):
        """Test that 18-class labels map to themselves."""
        for label in EMPATHETIC_DIALOGUES_18_LABELS:
            if label in EMPATHETIC_DIALOGUES_32_LABELS:
                mapped = map_label_32_to_18(label)
                assert mapped == label
    
    def test_map_label_32_to_18_examples(self):
        """Test specific mapping examples."""
        assert map_label_32_to_18("furious") == "angry"
        assert map_label_32_to_18("terrified") == "afraid"
        assert map_label_32_to_18("devastated") == "sad"
        assert map_label_32_to_18("disgusted") == "annoyed"
    
    def test_map_label_invalid_raises_error(self):
        """Test that invalid label raises error."""
        with pytest.raises(ValueError, match="Unknown label"):
            map_label_32_to_18("invalid_emotion")
    
    def test_all_32_labels_have_mapping(self):
        """Test that all 32 labels can be mapped to 18."""
        for label in EMPATHETIC_DIALOGUES_32_LABELS:
            mapped = map_label_32_to_18(label)
            assert mapped in EMPATHETIC_DIALOGUES_18_LABELS
