"""
Tests for EmpatheticDialogues data loader.
"""

import pytest
from pathlib import Path

from insideout.data.loader import EmpatheticDialoguesLoader
from insideout.core.models import Dialogue, DialogueTurn


# Use the actual data directory in the project
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "empatheticdialogues"


@pytest.fixture
def loader_32():
    """Fixture for loader with 32 classes."""
    return EmpatheticDialoguesLoader(DATA_DIR, label_set="32")


@pytest.fixture
def loader_18():
    """Fixture for loader with 18 classes."""
    return EmpatheticDialoguesLoader(DATA_DIR, label_set="18")


class TestEmpatheticDialoguesLoader:
    """Tests for EmpatheticDialoguesLoader."""
    
    def test_init_valid_label_set(self):
        """Test initialization with valid label sets."""
        loader_32 = EmpatheticDialoguesLoader(DATA_DIR, label_set="32")
        assert loader_32.label_set == "32"
        assert len(loader_32.label_list) == 32
        
        loader_18 = EmpatheticDialoguesLoader(DATA_DIR, label_set="18")
        assert loader_18.label_set == "18"
        assert len(loader_18.label_list) == 18
    
    def test_init_invalid_label_set(self):
        """Test initialization with invalid label set raises error."""
        with pytest.raises(ValueError, match="label_set must be"):
            EmpatheticDialoguesLoader(DATA_DIR, label_set="10")
    
    def test_init_invalid_data_dir(self):
        """Test initialization with non-existent directory raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            EmpatheticDialoguesLoader(Path("/nonexistent/path"), label_set="32")
    
    def test_load_split_test(self, loader_32):
        """Test loading test split."""
        dialogues = loader_32.load_split("test", limit=5)
        
        assert len(dialogues) == 5
        assert all(isinstance(d, Dialogue) for d in dialogues)
        assert all(len(d.turns) > 0 for d in dialogues)
    
    def test_load_split_train(self, loader_32):
        """Test loading train split."""
        dialogues = loader_32.load_split("train", limit=3)
        
        assert len(dialogues) == 3
        assert all(isinstance(d, Dialogue) for d in dialogues)
    
    def test_load_split_val(self, loader_32):
        """Test loading validation split."""
        dialogues = loader_32.load_split("val", limit=3)
        
        assert len(dialogues) == 3
        assert all(isinstance(d, Dialogue) for d in dialogues)
    
    def test_dialogue_structure(self, loader_32):
        """Test that loaded dialogues have correct structure."""
        dialogues = loader_32.load_split("test", limit=1)
        dialogue = dialogues[0]
        
        # Check required fields
        assert isinstance(dialogue.conv_id, str)
        assert len(dialogue.conv_id) > 0
        
        assert isinstance(dialogue.emotion_label, str)
        assert dialogue.emotion_label in loader_32.label_list
        
        assert isinstance(dialogue.prompt, str)
        assert len(dialogue.prompt) > 0
        
        assert isinstance(dialogue.turns, list)
        assert len(dialogue.turns) > 0
        
        # Check turns
        for turn in dialogue.turns:
            assert isinstance(turn, DialogueTurn)
            assert isinstance(turn.speaker_idx, int)
            assert turn.speaker_idx in [0, 1]
            assert isinstance(turn.utterance, str)
            assert len(turn.utterance) > 0
    
    def test_dialogue_history_str(self, loader_32):
        """Test dialogue history string formatting."""
        dialogues = loader_32.load_split("test", limit=1)
        dialogue = dialogues[0]
        
        history = dialogue.get_dialogue_history_str()
        
        assert isinstance(history, str)
        assert len(history) > 0
        assert "Speaker" in history
        
        # Test with max_turns
        if len(dialogue.turns) > 2:
            history_limited = dialogue.get_dialogue_history_str(max_turns=2)
            assert len(history_limited) < len(history)
    
    def test_label_mapping_18(self, loader_18):
        """Test that 32-class labels are mapped to 18-class."""
        dialogues = loader_18.load_split("test", limit=10)
        
        for dialogue in dialogues:
            # All labels should be from 18-class set
            assert dialogue.emotion_label in loader_18.label_list
    
    def test_utterance_decoding(self, loader_32):
        """Test that special characters are decoded correctly."""
        dialogues = loader_32.load_split("test", limit=20)
        
        # Check that no encoded characters remain
        for dialogue in dialogues:
            assert "_comma_" not in dialogue.prompt
            for turn in dialogue.turns:
                assert "_comma_" not in turn.utterance
    
    def test_limit_parameter(self, loader_32):
        """Test that limit parameter works correctly."""
        dialogues_5 = loader_32.load_split("test", limit=5)
        dialogues_10 = loader_32.load_split("test", limit=10)
        
        assert len(dialogues_5) == 5
        assert len(dialogues_10) == 10
        
        # First 5 should be the same
        for d5, d10 in zip(dialogues_5, dialogues_10):
            assert d5.conv_id == d10.conv_id
    
    def test_get_statistics(self, loader_32):
        """Test getting dataset statistics."""
        stats = loader_32.get_statistics("test")
        
        assert "split" in stats
        assert stats["split"] == "test"
        
        assert "total_dialogues" in stats
        assert stats["total_dialogues"] > 0
        
        assert "total_turns" in stats
        assert stats["total_turns"] > 0
        
        assert "avg_turns_per_dialogue" in stats
        assert stats["avg_turns_per_dialogue"] > 0
        
        assert "unique_labels" in stats
        assert "label_distribution" in stats
        
        # Check label distribution
        label_dist = stats["label_distribution"]
        assert isinstance(label_dist, dict)
        assert sum(label_dist.values()) == stats["total_dialogues"]


class TestDialogueModel:
    """Tests for Dialogue model validation."""
    
    def test_dialogue_with_empty_turns_raises_error(self):
        """Test that Dialogue with empty turns raises validation error."""
        with pytest.raises(ValueError):
            Dialogue(
                conv_id="test",
                turns=[],
                emotion_label="happy",
                prompt="test prompt"
            )
    
    def test_dialogue_turn_creation(self):
        """Test DialogueTurn creation."""
        turn = DialogueTurn(speaker_idx=1, utterance="Hello")
        
        assert turn.speaker_idx == 1
        assert turn.utterance == "Hello"
    
    def test_dialogue_creation(self):
        """Test Dialogue creation with valid data."""
        turns = [
            DialogueTurn(speaker_idx=1, utterance="I'm happy today"),
            DialogueTurn(speaker_idx=0, utterance="That's great!")
        ]
        
        dialogue = Dialogue(
            conv_id="test_conv",
            turns=turns,
            emotion_label="joyful",
            prompt="I had a great day"
        )
        
        assert dialogue.conv_id == "test_conv"
        assert len(dialogue.turns) == 2
        assert dialogue.emotion_label == "joyful"
        assert dialogue.prompt == "I had a great day"
