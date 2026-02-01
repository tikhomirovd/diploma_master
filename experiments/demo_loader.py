"""
Demo script to test the data loader.

This script loads a small sample from the dataset and displays information.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from insideout.data.loader import EmpatheticDialoguesLoader
from insideout.logging.logger import setup_logger


def main():
    """Run loader demo."""
    # Setup logger
    logger = setup_logger(
        log_dir=Path("logs"),
        experiment_name="demo_loader"
    )
    
    logger.info("Starting data loader demo")
    
    # Create loader
    data_dir = Path("data/empatheticdialogues")
    loader = EmpatheticDialoguesLoader(data_dir, label_set="32")
    
    logger.info(f"Loader initialized with {len(loader.label_list)} emotion labels")
    
    # Get statistics
    for split in ["train", "val", "test"]:
        stats = loader.get_statistics(split)
        logger.info(f"\n{split.upper()} Split Statistics:")
        logger.info(f"  Total dialogues: {stats['total_dialogues']}")
        logger.info(f"  Total turns: {stats['total_turns']}")
        logger.info(f"  Avg turns per dialogue: {stats['avg_turns_per_dialogue']:.2f}")
        logger.info(f"  Unique labels: {stats['unique_labels']}")
    
    # Load and display a few examples
    logger.info("\n" + "="*80)
    logger.info("Loading 3 example dialogues from test set:")
    logger.info("="*80)
    
    dialogues = loader.load_split("test", limit=3)
    
    for i, dialogue in enumerate(dialogues, 1):
        logger.info(f"\nDialogue {i}:")
        logger.info(f"  Conv ID: {dialogue.conv_id}")
        logger.info(f"  Emotion: {dialogue.emotion_label}")
        logger.info(f"  Prompt: {dialogue.prompt}")
        logger.info(f"  Number of turns: {len(dialogue.turns)}")
        logger.info(f"  Dialogue:")
        for turn in dialogue.turns:
            speaker = f"Speaker {turn.speaker_idx}"
            logger.info(f"    {speaker}: {turn.utterance}")
    
    # Test 18-class mapping
    logger.info("\n" + "="*80)
    logger.info("Testing 18-class label mapping:")
    logger.info("="*80)
    
    loader_18 = EmpatheticDialoguesLoader(data_dir, label_set="18")
    dialogues_18 = loader_18.load_split("test", limit=5)
    
    logger.info(f"Loaded {len(dialogues_18)} dialogues with 18-class labels:")
    for dialogue in dialogues_18:
        logger.info(f"  {dialogue.conv_id}: {dialogue.emotion_label}")
    
    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
