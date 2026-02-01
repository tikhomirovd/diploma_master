"""
Shared pytest fixtures for all tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
    # Reset cache initialization state
    import insideout.cache.llm_cache as cache_module
    cache_module._cache_initialized = False


@pytest.fixture
def temp_db_path(temp_dir):
    """Create a temporary database path."""
    return temp_dir / "test.db"


@pytest.fixture
def sample_dialogue_data():
    """Sample dialogue data for testing."""
    return {
        "conv_id": "test_conv_1",
        "turns": [
            {"speaker_idx": 0, "utterance": "How are you?"},
            {"speaker_idx": 1, "utterance": "I'm feeling great!"}
        ],
        "emotion_label": "joyful",
        "prompt": "I just got good news"
    }


@pytest.fixture
def sample_erc_output():
    """Sample ERC output for testing."""
    return {
        "agent_emotion": "happiness",
        "predicted_label": "joyful",
        "confidence": 0.85,
        "rationale": "The speaker expresses positive sentiment"
    }


@pytest.fixture
def sample_erg_candidate():
    """Sample ERG candidate for testing."""
    return {
        "agent_emotion": "happiness",
        "response": "That's wonderful! I'm happy for you!",
        "reasoning": "Positive reinforcement from happiness perspective"
    }
