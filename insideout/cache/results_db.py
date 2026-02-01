"""
SQLite database for storing experiment results.

Stores:
- Experiment configurations
- ERC predictions and metrics
- ERG generated responses and metrics
- Enables easy querying and analysis of results
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from insideout.core.models import (
    ExperimentConfig,
    ERCResult,
    ERGResult
)


logger = logging.getLogger("insideout.cache")


class ResultsDatabase:
    """SQLite database for storing experiment results."""
    
    def __init__(self, db_path: Path):
        """
        Initialize the results database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"Results database initialized at: {self.db_path}")
    
    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    model_provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    task_type TEXT NOT NULL,
                    method TEXT NOT NULL,
                    label_set TEXT NOT NULL,
                    split TEXT NOT NULL,
                    sample_limit INTEGER,
                    config_json TEXT NOT NULL
                )
            """)
            
            # ERC results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS erc_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    conv_id TEXT NOT NULL,
                    ground_truth TEXT NOT NULL,
                    predicted TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rationale TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            
            # ERG results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS erg_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    conv_id TEXT NOT NULL,
                    reference_response TEXT NOT NULL,
                    generated_response TEXT NOT NULL,
                    predicted_emotion TEXT,
                    metrics_json TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_name 
                ON experiments(experiment_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_erc_experiment 
                ON erc_results(experiment_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_erg_experiment 
                ON erg_results(experiment_id)
            """)
            
            conn.commit()
    
    def save_experiment(self, config: ExperimentConfig) -> int:
        """
        Save experiment configuration to database.
        
        Args:
            config: Experiment configuration
        
        Returns:
            experiment_id: ID of the saved experiment
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            config_json = config.model_dump_json()
            
            cursor.execute("""
                INSERT INTO experiments (
                    timestamp, experiment_name, model_provider, model_name,
                    temperature, task_type, method, label_set, split, sample_limit,
                    config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                config.experiment_name,
                config.model_provider,
                config.model_name,
                config.temperature,
                config.task_type,
                config.method,
                config.label_set,
                config.split,
                config.limit,
                config_json
            ))
            
            experiment_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Saved experiment config: id={experiment_id}, name={config.experiment_name}")
            return experiment_id
    
    def save_erc_results(
        self,
        experiment_id: int,
        results: List[ERCResult]
    ) -> None:
        """
        Save ERC results to database.
        
        Args:
            experiment_id: ID of the experiment
            results: List of ERC results
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute("""
                    INSERT INTO erc_results (
                        experiment_id, conv_id, ground_truth, predicted,
                        confidence, rationale
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    result.conv_id,
                    result.ground_truth,
                    result.predicted,
                    result.confidence,
                    result.rationale
                ))
            
            conn.commit()
            logger.info(f"Saved {len(results)} ERC results for experiment {experiment_id}")
    
    def save_erg_results(
        self,
        experiment_id: int,
        results: List[ERGResult],
        metrics: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Save ERG results to database.
        
        Args:
            experiment_id: ID of the experiment
            results: List of ERG results
            metrics: Optional list of metrics dicts (one per result)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for i, result in enumerate(results):
                metrics_json = None
                if metrics and i < len(metrics):
                    metrics_json = json.dumps(metrics[i])
                
                cursor.execute("""
                    INSERT INTO erg_results (
                        experiment_id, conv_id, reference_response,
                        generated_response, predicted_emotion, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    result.conv_id,
                    result.reference_response,
                    result.generated_response,
                    result.predicted_emotion,
                    metrics_json
                ))
            
            conn.commit()
            logger.info(f"Saved {len(results)} ERG results for experiment {experiment_id}")
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve experiment configuration by ID.
        
        Args:
            experiment_id: ID of the experiment
        
        Returns:
            Experiment data as dictionary, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM experiments WHERE id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_erc_results(
        self,
        experiment_id: int
    ) -> List[ERCResult]:
        """
        Retrieve ERC results for an experiment.
        
        Args:
            experiment_id: ID of the experiment
        
        Returns:
            List of ERC results
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT conv_id, ground_truth, predicted, confidence, rationale
                FROM erc_results
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append(ERCResult(
                    conv_id=row["conv_id"],
                    ground_truth=row["ground_truth"],
                    predicted=row["predicted"],
                    confidence=row["confidence"],
                    rationale=row["rationale"]
                ))
            
            return results
    
    def get_erg_results(
        self,
        experiment_id: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ERG results for an experiment.
        
        Args:
            experiment_id: ID of the experiment
        
        Returns:
            List of ERG result dictionaries (includes metrics)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT conv_id, reference_response, generated_response,
                       predicted_emotion, metrics_json
                FROM erg_results
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    "conv_id": row["conv_id"],
                    "reference_response": row["reference_response"],
                    "generated_response": row["generated_response"],
                    "predicted_emotion": row["predicted_emotion"]
                }
                
                if row["metrics_json"]:
                    result["metrics"] = json.loads(row["metrics_json"])
                
                results.append(result)
            
            return results
    
    def list_experiments(
        self,
        task_type: Optional[str] = None,
        method: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List recent experiments.
        
        Args:
            task_type: Filter by task type ('erc' or 'erg')
            method: Filter by method ('baseline' or 'insideout')
            limit: Maximum number of experiments to return
        
        Returns:
            List of experiment dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM experiments WHERE 1=1"
            params = []
            
            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)
            
            if method:
                query += " AND method = ?"
                params.append(method)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append(dict(row))
            
            return experiments
    
    def delete_experiment(self, experiment_id: int) -> None:
        """
        Delete an experiment and all its results.
        
        Args:
            experiment_id: ID of the experiment to delete
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete results first (due to foreign key constraints)
            cursor.execute("DELETE FROM erc_results WHERE experiment_id = ?", (experiment_id,))
            cursor.execute("DELETE FROM erg_results WHERE experiment_id = ?", (experiment_id,))
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            
            conn.commit()
            logger.info(f"Deleted experiment {experiment_id} and all its results")


def create_results_db(db_path: Path) -> ResultsDatabase:
    """
    Convenience function to create a results database.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        ResultsDatabase instance
    """
    return ResultsDatabase(db_path)
