"""
Tests for Results Database.
"""

import pytest
import sqlite3
from pathlib import Path

from insideout.cache import ResultsDatabase, create_results_db
from insideout.core.models import ExperimentConfig, ERCResult, ERGResult


class TestResultsDatabaseInitialization:
    """Tests for database initialization and connection."""
    
    def test_database_creation(self, temp_db_path):
        """Test that database file is created."""
        db = ResultsDatabase(temp_db_path)
        assert temp_db_path.exists()
    
    def test_database_in_nested_directory(self, temp_dir):
        """Test database creation in nested directory."""
        nested_path = temp_dir / "level1" / "level2" / "test.db"
        db = ResultsDatabase(nested_path)
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    def test_database_tables_created(self, temp_db_path):
        """Test that all required tables are created."""
        db = ResultsDatabase(temp_db_path)
        
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            
            assert "experiments" in tables
            assert "erc_results" in tables
            assert "erg_results" in tables
    
    def test_database_schema_experiments(self, temp_db_path):
        """Test experiments table schema."""
        db = ResultsDatabase(temp_db_path)
        
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(experiments)")
            columns = {row[1] for row in cursor.fetchall()}
            
            required_columns = {
                "id", "timestamp", "experiment_name", "model_provider",
                "model_name", "temperature", "task_type", "method",
                "label_set", "split", "sample_limit", "config_json"
            }
            
            assert required_columns.issubset(columns)
    
    def test_database_schema_erc_results(self, temp_db_path):
        """Test erc_results table schema."""
        db = ResultsDatabase(temp_db_path)
        
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(erc_results)")
            columns = {row[1] for row in cursor.fetchall()}
            
            required_columns = {
                "id", "experiment_id", "conv_id", "ground_truth",
                "predicted", "confidence", "rationale"
            }
            
            assert required_columns.issubset(columns)
    
    def test_database_schema_erg_results(self, temp_db_path):
        """Test erg_results table schema."""
        db = ResultsDatabase(temp_db_path)
        
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(erg_results)")
            columns = {row[1] for row in cursor.fetchall()}
            
            required_columns = {
                "id", "experiment_id", "conv_id", "reference_response",
                "generated_response", "predicted_emotion", "metrics_json"
            }
            
            assert required_columns.issubset(columns)
    
    def test_database_indices_created(self, temp_db_path):
        """Test that indices are created."""
        db = ResultsDatabase(temp_db_path)
        
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indices = {row[0] for row in cursor.fetchall()}
            
            assert "idx_experiments_name" in indices
            assert "idx_erc_experiment" in indices
            assert "idx_erg_experiment" in indices
    
    def test_database_connection_works(self, temp_db_path):
        """Test that we can connect and query the database."""
        db = ResultsDatabase(temp_db_path)
        
        # Should be able to query without errors
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiments")
            count = cursor.fetchone()[0]
            assert count == 0


class TestResultsDatabaseExperiments:
    """Tests for experiment operations."""
    
    def test_save_experiment(self, temp_db_path):
        """Test saving experiment configuration."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="test_experiment",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="insideout",
            label_set="32",
            split="test",
            limit=10
        )
        
        experiment_id = db.save_experiment(config)
        assert experiment_id > 0
    
    def test_get_experiment(self, temp_db_path):
        """Test retrieving experiment configuration."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="test_exp",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        
        experiment_id = db.save_experiment(config)
        retrieved = db.get_experiment(experiment_id)
        
        assert retrieved is not None
        assert retrieved["experiment_name"] == "test_exp"
        assert retrieved["model_provider"] == "openai"
        assert retrieved["task_type"] == "erc"
        assert retrieved["method"] == "baseline"
    
    def test_get_nonexistent_experiment(self, temp_db_path):
        """Test getting experiment that doesn't exist."""
        db = ResultsDatabase(temp_db_path)
        retrieved = db.get_experiment(99999)
        assert retrieved is None
    
    def test_save_multiple_experiments(self, temp_db_path):
        """Test saving multiple experiments."""
        db = ResultsDatabase(temp_db_path)
        
        ids = []
        for i in range(5):
            config = ExperimentConfig(
                experiment_name=f"exp_{i}",
                model_provider="openai",
                model_name="gpt-4",
                temperature=0.0,
                task_type="erc",
                method="baseline",
                label_set="32",
                split="test"
            )
            ids.append(db.save_experiment(config))
        
        # All IDs should be unique
        assert len(set(ids)) == 5
        
        # All should be retrievable
        for exp_id in ids:
            assert db.get_experiment(exp_id) is not None


class TestResultsDatabaseERCResults:
    """Tests for ERC results operations."""
    
    def test_save_erc_results(self, temp_db_path):
        """Test saving ERC results."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="erc_test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERCResult(
                conv_id="conv_1",
                ground_truth="happy",
                predicted="happy",
                confidence=0.9,
                rationale="Clear positive sentiment"
            ),
            ERCResult(
                conv_id="conv_2",
                ground_truth="sad",
                predicted="angry",
                confidence=0.7,
                rationale="Negative tone detected"
            )
        ]
        
        # Should not raise
        db.save_erc_results(experiment_id, results)
    
    def test_get_erc_results(self, temp_db_path):
        """Test retrieving ERC results."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="erc_test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERCResult(
                conv_id="conv_1",
                ground_truth="happy",
                predicted="happy",
                confidence=0.9
            )
        ]
        db.save_erc_results(experiment_id, results)
        
        retrieved = db.get_erc_results(experiment_id)
        
        assert len(retrieved) == 1
        assert retrieved[0].conv_id == "conv_1"
        assert retrieved[0].is_correct
    
    def test_get_erc_results_empty(self, temp_db_path):
        """Test getting ERC results when none exist."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="empty_test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        retrieved = db.get_erc_results(experiment_id)
        assert len(retrieved) == 0
    
    def test_erc_results_is_correct_property(self, temp_db_path):
        """Test ERCResult is_correct property."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERCResult(conv_id="c1", ground_truth="happy", predicted="happy", confidence=0.9),
            ERCResult(conv_id="c2", ground_truth="sad", predicted="angry", confidence=0.8)
        ]
        db.save_erc_results(experiment_id, results)
        
        retrieved = db.get_erc_results(experiment_id)
        assert retrieved[0].is_correct is True
        assert retrieved[1].is_correct is False


class TestResultsDatabaseERGResults:
    """Tests for ERG results operations."""
    
    def test_save_erg_results(self, temp_db_path):
        """Test saving ERG results."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="erg_test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erg",
            method="insideout",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERGResult(
                conv_id="conv_1",
                reference_response="That sounds wonderful!",
                generated_response="I'm so happy to hear that!",
                predicted_emotion="happy"
            )
        ]
        
        metrics = [{"bleu-1": 0.5, "rouge-1": 0.6}]
        
        db.save_erg_results(experiment_id, results, metrics)
    
    def test_get_erg_results(self, temp_db_path):
        """Test retrieving ERG results."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="erg_test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erg",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERGResult(
                conv_id="conv_1",
                reference_response="Great!",
                generated_response="Wonderful!",
                predicted_emotion="happy"
            )
        ]
        metrics = [{"bleu-1": 0.7}]
        
        db.save_erg_results(experiment_id, results, metrics)
        
        retrieved = db.get_erg_results(experiment_id)
        
        assert len(retrieved) == 1
        assert retrieved[0]["conv_id"] == "conv_1"
        assert retrieved[0]["predicted_emotion"] == "happy"
        assert "metrics" in retrieved[0]
        assert retrieved[0]["metrics"]["bleu-1"] == 0.7
    
    def test_erg_results_without_metrics(self, temp_db_path):
        """Test saving ERG results without metrics."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erg",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERGResult(
                conv_id="conv_1",
                reference_response="Great!",
                generated_response="Wonderful!"
            )
        ]
        
        db.save_erg_results(experiment_id, results)
        
        retrieved = db.get_erg_results(experiment_id)
        assert len(retrieved) == 1
        # metrics key might not exist or be None
        assert "metrics" not in retrieved[0] or retrieved[0].get("metrics") is None


class TestResultsDatabaseQueries:
    """Tests for database query operations."""
    
    def test_list_experiments_all(self, temp_db_path):
        """Test listing all experiments."""
        db = ResultsDatabase(temp_db_path)
        
        for i in range(3):
            config = ExperimentConfig(
                experiment_name=f"exp_{i}",
                model_provider="openai",
                model_name="gpt-4",
                temperature=0.0,
                task_type="erc",
                method="baseline",
                label_set="32",
                split="test"
            )
            db.save_experiment(config)
        
        experiments = db.list_experiments()
        assert len(experiments) == 3
    
    def test_list_experiments_by_task_type(self, temp_db_path):
        """Test listing experiments filtered by task type."""
        db = ResultsDatabase(temp_db_path)
        
        for i in range(2):
            config = ExperimentConfig(
                experiment_name=f"erc_{i}",
                model_provider="openai",
                model_name="gpt-4",
                temperature=0.0,
                task_type="erc",
                method="baseline",
                label_set="32",
                split="test"
            )
            db.save_experiment(config)
        
        config = ExperimentConfig(
            experiment_name="erg_1",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erg",
            method="baseline",
            label_set="32",
            split="test"
        )
        db.save_experiment(config)
        
        erc_exps = db.list_experiments(task_type="erc")
        assert len(erc_exps) == 2
        
        erg_exps = db.list_experiments(task_type="erg")
        assert len(erg_exps) == 1
    
    def test_list_experiments_by_method(self, temp_db_path):
        """Test listing experiments filtered by method."""
        db = ResultsDatabase(temp_db_path)
        
        config1 = ExperimentConfig(
            experiment_name="baseline_exp",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        db.save_experiment(config1)
        
        config2 = ExperimentConfig(
            experiment_name="insideout_exp",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="insideout",
            label_set="32",
            split="test"
        )
        db.save_experiment(config2)
        
        baseline_exps = db.list_experiments(method="baseline")
        assert len(baseline_exps) == 1
        assert baseline_exps[0]["method"] == "baseline"
    
    def test_list_experiments_with_limit(self, temp_db_path):
        """Test listing experiments with limit."""
        db = ResultsDatabase(temp_db_path)
        
        for i in range(10):
            config = ExperimentConfig(
                experiment_name=f"exp_{i}",
                model_provider="openai",
                model_name="gpt-4",
                temperature=0.0,
                task_type="erc",
                method="baseline",
                label_set="32",
                split="test"
            )
            db.save_experiment(config)
        
        experiments = db.list_experiments(limit=5)
        assert len(experiments) == 5
    
    def test_list_experiments_empty(self, temp_db_path):
        """Test listing experiments when none exist."""
        db = ResultsDatabase(temp_db_path)
        experiments = db.list_experiments()
        assert len(experiments) == 0


class TestResultsDatabaseDelete:
    """Tests for delete operations."""
    
    def test_delete_experiment(self, temp_db_path):
        """Test deleting experiment and its results."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="to_delete",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERCResult(
                conv_id="conv_1",
                ground_truth="happy",
                predicted="happy",
                confidence=0.9
            )
        ]
        db.save_erc_results(experiment_id, results)
        
        # Verify it exists
        assert db.get_experiment(experiment_id) is not None
        assert len(db.get_erc_results(experiment_id)) == 1
        
        # Delete
        db.delete_experiment(experiment_id)
        
        # Verify it's gone
        assert db.get_experiment(experiment_id) is None
        assert len(db.get_erc_results(experiment_id)) == 0
    
    def test_delete_nonexistent_experiment(self, temp_db_path):
        """Test deleting experiment that doesn't exist."""
        db = ResultsDatabase(temp_db_path)
        
        # Should not raise error
        db.delete_experiment(99999)


class TestResultsDatabaseConvenience:
    """Tests for convenience functions."""
    
    def test_create_results_db(self, temp_db_path):
        """Test create_results_db convenience function."""
        db = create_results_db(temp_db_path)
        
        assert isinstance(db, ResultsDatabase)
        assert temp_db_path.exists()
    
    def test_database_path_property(self, temp_db_path):
        """Test that database path is accessible."""
        db = ResultsDatabase(temp_db_path)
        assert db.db_path == temp_db_path


class TestResultsDatabaseEdgeCases:
    """Edge case tests for Results Database."""
    
    def test_very_long_experiment_name(self, temp_db_path):
        """Test with very long experiment name."""
        db = ResultsDatabase(temp_db_path)
        
        long_name = "a" * 1000
        config = ExperimentConfig(
            experiment_name=long_name,
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        
        experiment_id = db.save_experiment(config)
        retrieved = db.get_experiment(experiment_id)
        assert retrieved["experiment_name"] == long_name
    
    def test_special_characters_in_names(self, temp_db_path):
        """Test with special characters."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="test_exp_!@#$%^&*()",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        
        experiment_id = db.save_experiment(config)
        assert db.get_experiment(experiment_id) is not None
    
    def test_unicode_in_results(self, temp_db_path):
        """Test with Unicode characters."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="unicode_test",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        results = [
            ERCResult(
                conv_id="conv_1",
                ground_truth="счастливый",  # Russian
                predicted="幸せ",  # Japanese
                confidence=0.9,
                rationale="Тестирование Unicode 测试"
            )
        ]
        
        db.save_erc_results(experiment_id, results)
        retrieved = db.get_erc_results(experiment_id)
        
        assert retrieved[0].ground_truth == "счастливый"
        assert retrieved[0].predicted == "幸せ"
    
    def test_empty_results_list(self, temp_db_path):
        """Test saving empty results list."""
        db = ResultsDatabase(temp_db_path)
        
        config = ExperimentConfig(
            experiment_name="empty_results",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            task_type="erc",
            method="baseline",
            label_set="32",
            split="test"
        )
        experiment_id = db.save_experiment(config)
        
        # Should not raise
        db.save_erc_results(experiment_id, [])
        
        retrieved = db.get_erc_results(experiment_id)
        assert len(retrieved) == 0
