"""
Tests for prompt templates.
"""

import pytest

from insideout.core.emotions import EmotionType
from insideout.core.prompts import (
    format_erc_emotional_prompt,
    format_erc_aggregate_prompt,
    format_erc_baseline_prompt,
    format_erg_zeroshot_prompt,
    format_erg_emotional_prompt,
    format_erg_aggregate_prompt,
    format_erg_baseline_prompt
)


class TestERCPrompts:
    """Tests for ERC prompt formatting."""
    
    def test_format_erc_emotional_prompt(self):
        """Test formatting emotional agent prompt for ERC."""
        prompt = format_erc_emotional_prompt(
            emotion=EmotionType.ANGER,
            dialogue_history="Speaker 0: How are you?\nSpeaker 1: I'm frustrated!",
            context="Dealing with work stress",
            speaker_idx=1,
            label_list=["angry", "sad", "happy"]
        )
        
        assert "anger" in prompt.lower()
        assert "Speaker 1: I'm frustrated!" in prompt
        assert "Dealing with work stress" in prompt
        assert "angry, sad, happy" in prompt
        assert "JSON" in prompt
        assert "predicted_label" in prompt
        assert "confidence" in prompt
        assert "rationale" in prompt
    
    def test_format_erc_aggregate_prompt(self):
        """Test formatting aggregate agent prompt for ERC."""
        agent_outputs = [
            {
                "agent_emotion": "anger",
                "predicted_label": "angry",
                "confidence": 0.8,
                "rationale": "Clear signs of anger"
            },
            {
                "agent_emotion": "happiness",
                "predicted_label": "happy",
                "confidence": 0.3,
                "rationale": "Some positive words"
            }
        ]
        
        prompt = format_erc_aggregate_prompt(agent_outputs)
        
        assert "aggregate agent" in prompt.lower()
        assert "anger" in prompt.lower()
        assert "happiness" in prompt.lower()
        assert "angry" in prompt
        assert "0.80" in prompt
        assert "Clear signs of anger" in prompt
        assert "JSON" in prompt
        assert "final_label" in prompt
    
    def test_format_erc_baseline_prompt(self):
        """Test formatting baseline prompt for ERC."""
        prompt = format_erc_baseline_prompt(
            dialogue_history="Speaker 0: What's wrong?\nSpeaker 1: Nothing!",
            context="Hiding emotions",
            speaker_idx=1,
            label_list=["angry", "sad", "neutral"]
        )
        
        assert "Speaker 1: Nothing!" in prompt
        assert "Hiding emotions" in prompt
        assert "angry, sad, neutral" in prompt
        assert "JSON" in prompt


class TestERGPrompts:
    """Tests for ERG prompt formatting."""
    
    def test_format_erg_zeroshot_prompt(self):
        """Test formatting zero-shot emotion estimation prompt."""
        prompt = format_erg_zeroshot_prompt(
            dialogue_history="Speaker 0: Tell me more.\nSpeaker 1: I'm so excited!",
            context="Good news received",
            label_list=["happy", "excited", "anxious"]
        )
        
        assert "Speaker 1: I'm so excited!" in prompt
        assert "Good news received" in prompt
        assert "happy, excited, anxious" in prompt
        assert "JSON" in prompt
        assert "predicted_emotion" in prompt
    
    def test_format_erg_emotional_prompt(self):
        """Test formatting emotional agent prompt for ERG."""
        prompt = format_erg_emotional_prompt(
            emotion=EmotionType.HAPPINESS,
            dialogue_history="Speaker 0: How was your day?\nSpeaker 1: It was rough.",
            context="Bad day at work",
            predicted_emotion="sad"
        )
        
        assert "happiness" in prompt.lower()
        assert "Speaker 1: It was rough." in prompt
        assert "Bad day at work" in prompt
        assert "sad" in prompt
        assert "empathetic response" in prompt.lower()
        assert "JSON" in prompt
        assert "response" in prompt
        assert "reasoning" in prompt
    
    def test_format_erg_aggregate_prompt(self):
        """Test formatting aggregate agent prompt for ERG."""
        candidates = [
            {
                "agent_emotion": "anger",
                "response": "I understand your frustration.",
                "reasoning": "Validation approach"
            },
            {
                "agent_emotion": "happiness",
                "response": "Things will get better!",
                "reasoning": "Optimistic support"
            }
        ]
        
        prompt = format_erg_aggregate_prompt(
            predicted_emotion="sad",
            dialogue_history="Speaker 0: What happened?\nSpeaker 1: Everything went wrong.",
            candidates=candidates
        )
        
        assert "aggregate agent" in prompt.lower()
        assert "sad" in prompt
        assert "Speaker 1: Everything went wrong." in prompt
        assert "I understand your frustration." in prompt
        assert "Things will get better!" in prompt
        assert "anger agent" in prompt.lower()
        assert "happiness agent" in prompt.lower()
        assert "JSON" in prompt
        assert "selected_response" in prompt
    
    def test_format_erg_baseline_prompt(self):
        """Test formatting baseline prompt for ERG."""
        prompt = format_erg_baseline_prompt(
            dialogue_history="Speaker 0: Are you okay?\nSpeaker 1: Not really.",
            context="Feeling down"
        )
        
        assert "Speaker 1: Not really." in prompt
        assert "Feeling down" in prompt
        assert "empathetic response" in prompt.lower()
        assert "JSON" in prompt


class TestPromptConsistency:
    """Tests for prompt consistency and format."""
    
    def test_all_prompts_request_json(self):
        """Test that all prompts request JSON output."""
        prompts = [
            format_erc_emotional_prompt(
                EmotionType.ANGER, "test", "test", 0, ["test"]
            ),
            format_erc_aggregate_prompt([{
                "agent_emotion": "anger",
                "predicted_label": "angry",
                "confidence": 0.5,
                "rationale": "test"
            }]),
            format_erc_baseline_prompt("test", "test", 0, ["test"]),
            format_erg_zeroshot_prompt("test", "test", ["test"]),
            format_erg_emotional_prompt(EmotionType.HAPPINESS, "test", "test", "happy"),
            format_erg_aggregate_prompt("happy", "test", [{
                "agent_emotion": "happiness",
                "response": "test",
                "reasoning": "test"
            }]),
            format_erg_baseline_prompt("test", "test")
        ]
        
        for prompt in prompts:
            assert "JSON" in prompt or "json" in prompt.lower()
            assert "{" in prompt  # JSON example
    
    def test_prompts_include_context(self):
        """Test that prompts properly include context information."""
        dialogue = "Speaker 0: Hi\nSpeaker 1: Hello"
        context = "Test context"
        
        # Test ERC prompts
        erc_emotional = format_erc_emotional_prompt(
            EmotionType.ANGER, dialogue, context, 1, ["test"]
        )
        assert dialogue in erc_emotional
        assert context in erc_emotional
        
        erc_baseline = format_erc_baseline_prompt(
            dialogue, context, 1, ["test"]
        )
        assert dialogue in erc_baseline
        assert context in erc_baseline
        
        # Test ERG prompts
        erg_zeroshot = format_erg_zeroshot_prompt(dialogue, context, ["test"])
        assert dialogue in erg_zeroshot
        assert context in erg_zeroshot
        
        erg_emotional = format_erg_emotional_prompt(
            EmotionType.HAPPINESS, dialogue, context, "happy"
        )
        assert dialogue in erg_emotional
        assert context in erg_emotional
