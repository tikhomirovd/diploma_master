"""
Prompt templates for ERC and ERG tasks.

Based on the InsideOut paper methodology:
- Emotional agents: agents embodying specific emotions
- Aggregate agents: consolidate outputs from emotional agents
- Zero-shot emotion estimation: for ERG pipeline
"""

from typing import List, Dict, Any
from string import Template

from insideout.core.emotions import EmotionType


# ============================================================================
# ERC (Emotion Recognition in Conversation) Prompts
# ============================================================================

ERC_EMOTIONAL_AGENT_TEMPLATE = Template("""You are an emotional agent embodying ${emotion_name}.
Your task is to analyze the following dialogue and identify the emotion of Speaker ${speaker_idx}.

Dialogue:
${dialogue_history}

Context: ${context}

From your ${emotion_name} perspective, what emotion is the speaker experiencing?
Provide your analysis in the following JSON format:
{
    "predicted_label": "emotion_label",
    "confidence": 0.X,
    "rationale": "your reasoning from the ${emotion_name} lens (2-3 sentences)"
}

Available emotion labels: ${label_list}

Return ONLY the JSON object, nothing else.""")


ERC_AGGREGATE_AGENT_TEMPLATE = Template("""You are an aggregate agent responsible for consolidating emotion predictions from multiple specialized emotional agents.

You have received predictions from 5 emotional agents (anger, disgust, fear, happiness, sadness):

${agent_outputs}

Your task is to consolidate these predictions into a single final judgment.

Provide your analysis in the following JSON format:
{
    "final_label": "emotion_label",
    "final_confidence": 0.X,
    "final_rationale": "synthesized reasoning combining all agents' perspectives (2-3 sentences)"
}

Consider:
- Consensus among agents
- Confidence levels of each agent
- Quality of rationales provided
- Context of the dialogue

Return ONLY the JSON object, nothing else.""")


ERC_BASELINE_TEMPLATE = Template("""Analyze the following dialogue and predict the speaker's emotion.

Dialogue:
${dialogue_history}

Context: ${context}

What emotion is Speaker ${speaker_idx} experiencing?
Choose from: ${label_list}

Provide your analysis in the following JSON format:
{
    "predicted_label": "emotion_label",
    "confidence": 0.X,
    "rationale": "your reasoning (2-3 sentences)"
}

Return ONLY the JSON object, nothing else.""")


# ============================================================================
# ERG (Empathetic Response Generation) Prompts
# ============================================================================

ERG_ZEROSHOT_EMOTION_TEMPLATE = Template("""Analyze the following dialogue and predict the speaker's emotion.

Dialogue:
${dialogue_history}

Context: ${context}

What emotion is the speaker experiencing?
Choose from: ${label_list}

Provide your analysis in the following JSON format:
{
    "predicted_emotion": "emotion_label",
    "confidence": 0.X,
    "reasoning": "brief explanation (1-2 sentences)"
}

Return ONLY the JSON object, nothing else.""")


ERG_EMOTIONAL_AGENT_TEMPLATE = Template("""You are an emotional agent embodying ${emotion_name}.
The speaker in this dialogue is currently experiencing: ${predicted_emotion}

Dialogue:
${dialogue_history}

Context: ${context}

From your ${emotion_name} perspective, generate an empathetic response aimed at improving the speaker's well-being and emotional state.

Provide your response in the following JSON format:
{
    "response": "your empathetic response text",
    "reasoning": "why this response is appropriate from the ${emotion_name} perspective (1-2 sentences)"
}

Your response should:
- Show understanding of the speaker's emotional state
- Provide emotional support or validation
- Offer helpful suggestions if appropriate
- Be natural and conversational

Return ONLY the JSON object, nothing else.""")


ERG_AGGREGATE_AGENT_TEMPLATE = Template("""You are an aggregate agent responsible for selecting the most empathetic and effective response.

Speaker's predicted emotion: ${predicted_emotion}

Dialogue context:
${dialogue_history}

Candidate responses from emotional agents:
${candidates}

Your task is to select the response that would be most effective for empathy and improving the speaker's well-being.

Provide your selection in the following JSON format:
{
    "selected_response": "the full text of the selected response",
    "reasoning": "why this response is the most effective (2-3 sentences)"
}

Consider:
- How well the response addresses the speaker's emotional needs
- Naturalness and fluency
- Appropriateness of suggestions or advice
- Emotional validation and support

Return ONLY the JSON object, nothing else.""")


ERG_BASELINE_TEMPLATE = Template("""Generate an empathetic response to the following dialogue.

Dialogue:
${dialogue_history}

Context: ${context}

Generate a response that:
- Shows understanding of the speaker's emotional state
- Provides emotional support or validation
- Offers helpful suggestions if appropriate
- Is natural and conversational

Provide your response in the following JSON format:
{
    "response": "your empathetic response text",
    "reasoning": "brief explanation of your approach (1-2 sentences)"
}

Return ONLY the JSON object, nothing else.""")


# ============================================================================
# Prompt Formatting Functions
# ============================================================================

def format_erc_emotional_prompt(
    emotion: EmotionType,
    dialogue_history: str,
    context: str,
    speaker_idx: int,
    label_list: List[str]
) -> str:
    """
    Format ERC prompt for an emotional agent.
    
    Args:
        emotion: The emotion this agent embodies
        dialogue_history: Formatted dialogue turns
        context: Situation/context prompt
        speaker_idx: Index of the speaker to analyze
        label_list: List of available emotion labels
    
    Returns:
        Formatted prompt string
    """
    return ERC_EMOTIONAL_AGENT_TEMPLATE.substitute(
        emotion_name=emotion.value,
        dialogue_history=dialogue_history,
        context=context,
        speaker_idx=speaker_idx,
        label_list=", ".join(label_list)
    )


def format_erc_aggregate_prompt(
    agent_outputs: List[Dict[str, Any]]
) -> str:
    """
    Format ERC prompt for aggregate agent.
    
    Args:
        agent_outputs: List of outputs from emotional agents
            Each should have: agent_emotion, predicted_label, confidence, rationale
    
    Returns:
        Formatted prompt string
    """
    # Format agent outputs for display
    outputs_text = []
    for i, output in enumerate(agent_outputs, 1):
        outputs_text.append(
            f"{i}. Agent: {output['agent_emotion']}\n"
            f"   Predicted: {output['predicted_label']}\n"
            f"   Confidence: {output['confidence']:.2f}\n"
            f"   Rationale: {output['rationale']}"
        )
    
    return ERC_AGGREGATE_AGENT_TEMPLATE.substitute(
        agent_outputs="\n\n".join(outputs_text)
    )


def format_erc_baseline_prompt(
    dialogue_history: str,
    context: str,
    speaker_idx: int,
    label_list: List[str]
) -> str:
    """
    Format ERC baseline prompt (direct LLM call).
    
    Args:
        dialogue_history: Formatted dialogue turns
        context: Situation/context prompt
        speaker_idx: Index of the speaker to analyze
        label_list: List of available emotion labels
    
    Returns:
        Formatted prompt string
    """
    return ERC_BASELINE_TEMPLATE.substitute(
        dialogue_history=dialogue_history,
        context=context,
        speaker_idx=speaker_idx,
        label_list=", ".join(label_list)
    )


def format_erg_zeroshot_prompt(
    dialogue_history: str,
    context: str,
    label_list: List[str]
) -> str:
    """
    Format ERG zero-shot emotion estimation prompt.
    
    Args:
        dialogue_history: Formatted dialogue turns
        context: Situation/context prompt
        label_list: List of available emotion labels
    
    Returns:
        Formatted prompt string
    """
    return ERG_ZEROSHOT_EMOTION_TEMPLATE.substitute(
        dialogue_history=dialogue_history,
        context=context,
        label_list=", ".join(label_list)
    )


def format_erg_emotional_prompt(
    emotion: EmotionType,
    dialogue_history: str,
    context: str,
    predicted_emotion: str
) -> str:
    """
    Format ERG prompt for an emotional agent.
    
    Args:
        emotion: The emotion this agent embodies
        dialogue_history: Formatted dialogue turns
        context: Situation/context prompt
        predicted_emotion: Emotion predicted by zero-shot step
    
    Returns:
        Formatted prompt string
    """
    return ERG_EMOTIONAL_AGENT_TEMPLATE.substitute(
        emotion_name=emotion.value,
        dialogue_history=dialogue_history,
        context=context,
        predicted_emotion=predicted_emotion
    )


def format_erg_aggregate_prompt(
    predicted_emotion: str,
    dialogue_history: str,
    candidates: List[Dict[str, Any]]
) -> str:
    """
    Format ERG prompt for aggregate agent.
    
    Args:
        predicted_emotion: Emotion predicted by zero-shot step
        dialogue_history: Formatted dialogue turns
        candidates: List of candidate responses from emotional agents
            Each should have: agent_emotion, response, reasoning
    
    Returns:
        Formatted prompt string
    """
    # Format candidates for display
    candidates_text = []
    for i, candidate in enumerate(candidates, 1):
        candidates_text.append(
            f"{i}. From {candidate['agent_emotion']} agent:\n"
            f"   Response: \"{candidate['response']}\"\n"
            f"   Reasoning: {candidate['reasoning']}"
        )
    
    return ERG_AGGREGATE_AGENT_TEMPLATE.substitute(
        predicted_emotion=predicted_emotion,
        dialogue_history=dialogue_history,
        candidates="\n\n".join(candidates_text)
    )


def format_erg_baseline_prompt(
    dialogue_history: str,
    context: str
) -> str:
    """
    Format ERG baseline prompt (direct LLM call).
    
    Args:
        dialogue_history: Formatted dialogue turns
        context: Situation/context prompt
    
    Returns:
        Formatted prompt string
    """
    return ERG_BASELINE_TEMPLATE.substitute(
        dialogue_history=dialogue_history,
        context=context
    )
