"""
Data loader for EmpatheticDialogues dataset.

Loads and parses CSV files, groups by conversation ID, and constructs Dialogue objects.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional

from insideout.core.models import Dialogue, DialogueTurn
from insideout.core.emotions import (
    get_labels_for_set,
    map_label_32_to_18,
)


class EmpatheticDialoguesLoader:
    """
    Loader for EmpatheticDialogues dataset.
    
    Loads dialogue data from CSV files, groups by conversation,
    and provides support for both 32-class and 18-class label sets.
    """
    
    def __init__(self, data_dir: Path, label_set: Literal["32", "18"] = "32"):
        """
        Initialize the loader.
        
        Args:
            data_dir: Path to directory containing train.csv, valid.csv, test.csv
            label_set: Either "32" for full label set or "18" for subset
        
        Raises:
            ValueError: If data_dir doesn't exist or label_set is invalid
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        if label_set not in ["32", "18"]:
            raise ValueError(f"label_set must be '32' or '18', got: {label_set}")
        
        self.label_set = label_set
        self.label_list = get_labels_for_set(label_set)
    
    def load_split(
        self,
        split: Literal["train", "val", "test"],
        limit: Optional[int] = None
    ) -> List[Dialogue]:
        """
        Load a specific split of the dataset.
        
        Args:
            split: Dataset split to load ("train", "val", or "test")
            limit: Optional limit on number of dialogues to load
        
        Returns:
            List of Dialogue objects
        
        Raises:
            FileNotFoundError: If split file doesn't exist
        """
        # Map 'val' to 'valid' for filename
        filename_split = "valid" if split == "val" else split
        csv_path = self.data_dir / f"{filename_split}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        
        # Read CSV and group by conversation ID
        conversations = self._load_csv_and_group(csv_path)
        
        # Convert to Dialogue objects
        dialogues = []
        for conv_id, turns_data in conversations.items():
            dialogue = self._create_dialogue(conv_id, turns_data)
            
            # Filter by label set if using 18 classes
            if self.label_set == "18":
                # Map label to 18-class set
                dialogue.emotion_label = map_label_32_to_18(dialogue.emotion_label)
            
            dialogues.append(dialogue)
            
            if limit and len(dialogues) >= limit:
                break
        
        return dialogues
    
    def _load_csv_and_group(self, csv_path: Path) -> Dict[str, List[dict]]:
        """
        Load CSV file and group rows by conversation ID.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Dictionary mapping conv_id to list of row dicts
        """
        conversations = defaultdict(list)
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv_id = row["conv_id"]
                conversations[conv_id].append(row)
        
        # Sort turns within each conversation by utterance_idx
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda x: int(x["utterance_idx"]))
        
        return conversations
    
    def _create_dialogue(self, conv_id: str, turns_data: List[dict]) -> Dialogue:
        """
        Create a Dialogue object from grouped conversation data.
        
        Args:
            conv_id: Conversation ID
            turns_data: List of turn dictionaries (sorted by utterance_idx)
        
        Returns:
            Dialogue object
        """
        # Extract context (emotion label) and prompt from first turn
        # All turns in a conversation share the same context and prompt
        first_turn = turns_data[0]
        emotion_label = first_turn["context"]
        prompt = first_turn["prompt"]
        
        # Parse utterance to handle special encoding (e.g., "_comma_")
        turns = []
        for turn_data in turns_data:
            utterance = self._decode_utterance(turn_data["utterance"])
            speaker_idx = int(turn_data["speaker_idx"])
            
            turn = DialogueTurn(
                speaker_idx=speaker_idx,
                utterance=utterance
            )
            turns.append(turn)
        
        return Dialogue(
            conv_id=conv_id,
            turns=turns,
            emotion_label=emotion_label,
            prompt=self._decode_utterance(prompt)
        )
    
    @staticmethod
    def _decode_utterance(text: str) -> str:
        """
        Decode special characters in utterance text.
        
        EmpatheticDialogues uses special encoding like "_comma_" for commas.
        
        Args:
            text: Encoded text
        
        Returns:
            Decoded text
        """
        # Replace special encodings
        replacements = {
            "_comma_": ",",
            "_period_": ".",
            "_exclamation_": "!",
            "_question_": "?",
            "_semicolon_": ";",
            "_colon_": ":",
            "_apostrophe_": "'",
            "_quote_": '"',
            "_dash_": "-",
            "_left_paren_": "(",
            "_right_paren_": ")",
        }
        
        decoded = text
        for encoded, decoded_char in replacements.items():
            decoded = decoded.replace(encoded, decoded_char)
        
        return decoded
    
    def get_statistics(self, split: Literal["train", "val", "test"]) -> dict:
        """
        Get statistics about a dataset split.
        
        Args:
            split: Dataset split
        
        Returns:
            Dictionary with statistics
        """
        dialogues = self.load_split(split)
        
        total_dialogues = len(dialogues)
        total_turns = sum(len(d.turns) for d in dialogues)
        avg_turns = total_turns / total_dialogues if total_dialogues > 0 else 0
        
        # Count labels
        label_counts = defaultdict(int)
        for dialogue in dialogues:
            label_counts[dialogue.emotion_label] += 1
        
        return {
            "split": split,
            "total_dialogues": total_dialogues,
            "total_turns": total_turns,
            "avg_turns_per_dialogue": avg_turns,
            "unique_labels": len(label_counts),
            "label_distribution": dict(label_counts),
        }


def get_loader(
    data_dir: Optional[Path] = None,
    label_set: Literal["32", "18"] = "32"
) -> EmpatheticDialoguesLoader:
    """
    Convenience function to get a loader with default data directory.
    
    Args:
        data_dir: Path to data directory (defaults to ./data/empatheticdialogues)
        label_set: Label set to use
    
    Returns:
        EmpatheticDialoguesLoader instance
    """
    if data_dir is None:
        data_dir = Path("data/empatheticdialogues")
    
    return EmpatheticDialoguesLoader(data_dir, label_set)
