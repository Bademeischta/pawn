"""
Archimedes Chess AI - Comprehensive Metrics Logger
Asynchronous SQLite logging for training, MCTS, chess-specific, and hardware metrics.
"""

import sqlite3
import threading
import queue
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import psutil

logger = logging.getLogger(__name__)

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class MetricsLogger:
    """
    Asynchronous metrics logger that writes to SQLite without blocking training.
    Captures training, MCTS, chess-specific, and hardware metrics.
    """
    
    ALLOWED_TABLES = {
        "training_metrics", "mcts_metrics", "chess_metrics",
        "hardware_metrics", "games", "position_analysis"
    }

    def __init__(self, db_path: str = "logs/training_logs.db", buffer_size: int = 500):
        self.db_path = Path(db_path)
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.queue = queue.Queue()
        self.running = True
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
            except:
                pass
        
        # Initialize database
        self._init_database()
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        logger.info(f"Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize all database tables."""
        conn = sqlite3.connect(str(self.db_path))
        
        # Enable WAL mode for better concurrency and performance
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        cursor = conn.cursor()
        
        # Training & Neural Network Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                batch INTEGER,
                loss_total REAL,
                loss_policy REAL,
                loss_value REAL,
                learning_rate REAL,
                gradient_norm REAL,
                accuracy_top1 REAL,
                accuracy_top5 REAL,
                train_loss REAL,
                val_loss REAL,
                overfitting_ratio REAL,
                epoch_duration REAL,
                samples_trained INTEGER,
                weight_mean REAL,
                weight_std REAL,
                activation_mean REAL,
                activation_std REAL
            )
        """)
        
        # MCTS & Search Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcts_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                avg_search_depth REAL,
                max_search_depth INTEGER,
                nodes_per_second REAL,
                branching_factor REAL,
                cutoff_rate REAL,
                cache_hit_rate REAL,
                q_value_mean REAL,
                q_value_std REAL,
                visit_count_mean REAL,
                visit_count_std REAL,
                puct_exploration REAL,
                puct_exploitation REAL,
                transposition_hits INTEGER,
                transposition_misses INTEGER
            )
        """)
        
        # Chess-Specific Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chess_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                elo_estimate REAL,
                win_rate REAL,
                loss_rate REAL,
                draw_rate REAL,
                win_by_mate REAL,
                win_by_time REAL,
                win_by_resignation REAL,
                draw_by_stalemate REAL,
                draw_by_fifty_moves REAL,
                draw_by_repetition REAL,
                white_performance REAL,
                black_performance REAL,
                avg_game_length REAL,
                blunder_rate REAL,
                centipawn_loss REAL,
                mate_in_x_accuracy REAL,
                opening_diversity REAL
            )
        """)
        
        # Hardware Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                gpu_utilization REAL,
                gpu_memory_used REAL,
                gpu_memory_total REAL,
                gpu_temperature REAL,
                cpu_percent REAL,
                ram_used REAL,
                ram_total REAL,
                disk_io_read REAL,
                disk_io_write REAL,
                positions_per_watt REAL
            )
        """)
        
        # Game Records (for PGN export)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                pgn TEXT,
                result TEXT,
                white_player TEXT,
                black_player TEXT,
                game_length INTEGER,
                opening_eco TEXT,
                final_fen TEXT
            )
        """)
        
        # Position Analysis (for visualization)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                fen TEXT,
                evaluation REAL,
                best_move TEXT,
                policy_distribution TEXT,
                attention_weights TEXT,
                feature_vector TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database schema initialized")
    
    def _writer_loop(self):
        """Background thread that writes metrics to database."""
        buffer = []
        
        while self.running or not self.queue.empty():
            try:
                # Get item with timeout
                item = self.queue.get(timeout=1.0)
                buffer.append(item)
                
                # Flush buffer if full or queue is empty
                if len(buffer) >= self.buffer_size or self.queue.empty():
                    self._flush_buffer(buffer)
                    buffer = []
                    
            except queue.Empty:
                # Flush any remaining items
                if buffer:
                    self._flush_buffer(buffer)
                    buffer = []
    
    def _flush_buffer(self, buffer: List[tuple]):
        """Write buffered metrics to database."""
        if not buffer:
            return
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Group by table
        tables = {}
        for table_name, data in buffer:
            # Security check for table name
            if table_name not in self.__class__.ALLOWED_TABLES:
                logger.error(f"Unauthorized table access blocked: {table_name}")
                continue

            if table_name not in tables:
                tables[table_name] = []
            tables[table_name].append(data)
        
        # Insert into each table
        for table_name, records in tables.items():
            if not records:
                continue
            
            # Build INSERT statement
            columns = list(records[0].keys())
            # Security check for column names (must be alphanumeric)
            columns = [c for c in columns if c.isidentifier()]

            placeholders = ','.join(['?' for _ in columns])
            query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
            
            # Convert records to tuples
            values = [tuple(record[col] for col in columns) for record in records]
            
            try:
                cursor.executemany(query, values)
            except sqlite3.Error as e:
                logger.error(f"Database insertion error in {table_name}: {e}")
        
        conn.commit()
        conn.close()
    
    def log_training(self, epoch: int, batch: int, metrics: Dict[str, Any]):
        """Log training and neural network metrics."""
        data = {
            'timestamp': time.time(),
            'epoch': epoch,
            'batch': batch,
            'loss_total': metrics.get('loss_total', 0.0),
            'loss_policy': metrics.get('loss_policy', 0.0),
            'loss_value': metrics.get('loss_value', 0.0),
            'learning_rate': metrics.get('learning_rate', 0.0),
            'gradient_norm': metrics.get('gradient_norm', 0.0),
            'accuracy_top1': metrics.get('accuracy_top1', 0.0),
            'accuracy_top5': metrics.get('accuracy_top5', 0.0),
            'train_loss': metrics.get('train_loss', 0.0),
            'val_loss': metrics.get('val_loss', 0.0),
            'overfitting_ratio': metrics.get('overfitting_ratio', 0.0),
            'epoch_duration': metrics.get('epoch_duration', 0.0),
            'samples_trained': metrics.get('samples_trained', 0),
            'weight_mean': metrics.get('weight_mean', 0.0),
            'weight_std': metrics.get('weight_std', 0.0),
            'activation_mean': metrics.get('activation_mean', 0.0),
            'activation_std': metrics.get('activation_std', 0.0),
        }
        self.queue.put(('training_metrics', data))
    
    def log_mcts(self, epoch: int, metrics: Dict[str, Any]):
        """Log MCTS and search metrics."""
        data = {
            'timestamp': time.time(),
            'epoch': epoch,
            'avg_search_depth': metrics.get('avg_search_depth', 0.0),
            'max_search_depth': metrics.get('max_search_depth', 0),
            'nodes_per_second': metrics.get('nodes_per_second', 0.0),
            'branching_factor': metrics.get('branching_factor', 0.0),
            'cutoff_rate': metrics.get('cutoff_rate', 0.0),
            'cache_hit_rate': metrics.get('cache_hit_rate', 0.0),
            'q_value_mean': metrics.get('q_value_mean', 0.0),
            'q_value_std': metrics.get('q_value_std', 0.0),
            'visit_count_mean': metrics.get('visit_count_mean', 0.0),
            'visit_count_std': metrics.get('visit_count_std', 0.0),
            'puct_exploration': metrics.get('puct_exploration', 0.0),
            'puct_exploitation': metrics.get('puct_exploitation', 0.0),
            'transposition_hits': metrics.get('transposition_hits', 0),
            'transposition_misses': metrics.get('transposition_misses', 0),
        }
        self.queue.put(('mcts_metrics', data))
    
    def log_chess(self, epoch: int, metrics: Dict[str, Any]):
        """Log chess-specific metrics."""
        data = {
            'timestamp': time.time(),
            'epoch': epoch,
            'elo_estimate': metrics.get('elo_estimate', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'loss_rate': metrics.get('loss_rate', 0.0),
            'draw_rate': metrics.get('draw_rate', 0.0),
            'win_by_mate': metrics.get('win_by_mate', 0.0),
            'win_by_time': metrics.get('win_by_time', 0.0),
            'win_by_resignation': metrics.get('win_by_resignation', 0.0),
            'draw_by_stalemate': metrics.get('draw_by_stalemate', 0.0),
            'draw_by_fifty_moves': metrics.get('draw_by_fifty_moves', 0.0),
            'draw_by_repetition': metrics.get('draw_by_repetition', 0.0),
            'white_performance': metrics.get('white_performance', 0.0),
            'black_performance': metrics.get('black_performance', 0.0),
            'avg_game_length': metrics.get('avg_game_length', 0.0),
            'blunder_rate': metrics.get('blunder_rate', 0.0),
            'centipawn_loss': metrics.get('centipawn_loss', 0.0),
            'mate_in_x_accuracy': metrics.get('mate_in_x_accuracy', 0.0),
            'opening_diversity': metrics.get('opening_diversity', 0.0),
        }
        self.queue.put(('chess_metrics', data))
    
    def log_hardware(self, epoch: int):
        """Log hardware metrics (CPU, RAM, GPU)."""
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        
        data = {
            'timestamp': time.time(),
            'epoch': epoch,
            'cpu_percent': cpu_percent,
            'ram_used': ram.used / (1024**3),  # GB
            'ram_total': ram.total / (1024**3),  # GB
            'disk_io_read': disk_io.read_bytes / (1024**2) if disk_io else 0.0,  # MB
            'disk_io_write': disk_io.write_bytes / (1024**2) if disk_io else 0.0,  # MB
            'gpu_utilization': 0.0,
            'gpu_memory_used': 0.0,
            'gpu_memory_total': 0.0,
            'gpu_temperature': 0.0,
            'positions_per_watt': 0.0,
        }
        
        # GPU metrics if available
        if self.gpu_available:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                
                data['gpu_utilization'] = util.gpu
                data['gpu_memory_used'] = mem_info.used / (1024**3)  # GB
                data['gpu_memory_total'] = mem_info.total / (1024**3)  # GB
                data['gpu_temperature'] = temp
            except:
                pass
        
        self.queue.put(('hardware_metrics', data))
    
    def log_game(self, epoch: int, pgn: str, result: str, white: str, black: str, 
                 game_length: int, opening_eco: str = "", final_fen: str = ""):
        """Log a complete game for PGN export."""
        data = {
            'timestamp': time.time(),
            'epoch': epoch,
            'pgn': pgn,
            'result': result,
            'white_player': white,
            'black_player': black,
            'game_length': game_length,
            'opening_eco': opening_eco,
            'final_fen': final_fen,
        }
        self.queue.put(('games', data))
    
    def log_position(self, epoch: int, fen: str, evaluation: float, best_move: str,
                     policy_dist: Dict[str, float], attention_weights: Optional[List[float]] = None,
                     feature_vector: Optional[List[float]] = None):
        """Log position analysis for visualization."""
        data = {
            'timestamp': time.time(),
            'epoch': epoch,
            'fen': fen,
            'evaluation': evaluation,
            'best_move': best_move,
            'policy_distribution': json.dumps(policy_dist),
            'attention_weights': json.dumps(attention_weights) if attention_weights else "[]",
            'feature_vector': json.dumps(feature_vector) if feature_vector else "[]",
        }
        self.queue.put(('position_analysis', data))
    
    def get_latest_metrics(self, table: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve latest metrics from a table."""
        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Unauthorized table access: {table}")

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Use parameterized query for limit, and validated table name
        cursor.execute(f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_metrics_range(self, table: str, start_epoch: int, end_epoch: int) -> List[Dict[str, Any]]:
        """Retrieve metrics for a specific epoch range."""
        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Unauthorized table access: {table}")

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            f"SELECT * FROM {table} WHERE epoch >= ? AND epoch <= ? ORDER BY timestamp",
            (start_epoch, end_epoch)
        )
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    
    def export_games_pgn(self, output_file: str, limit: Optional[int] = None):
        """Export games to PGN file."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT pgn FROM games ORDER BY timestamp DESC"
        params = ()
        if limit:
            query += " LIMIT ?"
            params = (limit,)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        with open(output_file, 'w') as f:
            for row in rows:
                f.write(row[0] + "\n\n")
        
        conn.close()
        logger.info(f"Exported {len(rows)} games to {output_file}")
    
    def close(self):
        """Gracefully shutdown the logger."""
        logger.info("Shutting down...")
        self.running = False

        # Wait for queue to be processed, but with a longer timeout if it has many items
        queue_size = self.queue.qsize()
        timeout = max(5.0, queue_size * 0.05)
        logger.info(f"Waiting up to {timeout:.1f}s for {queue_size} items to be flushed...")

        self.writer_thread.join(timeout=timeout)
        
        if self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        
        logger.info("Closed successfully")


if __name__ == "__main__":
    # Test the logger
    test_logger = MetricsLogger("test_metrics.db")
    
    # Simulate some metrics
    for epoch in range(3):
        test_logger.log_training(epoch, 0, {
            'loss_total': 0.5 - epoch * 0.1,
            'loss_policy': 0.3 - epoch * 0.05,
            'loss_value': 0.2 - epoch * 0.05,
            'learning_rate': 0.001,
            'accuracy_top1': 0.5 + epoch * 0.1,
        })
        
        test_logger.log_mcts(epoch, {
            'avg_search_depth': 10.0 + epoch,
            'nodes_per_second': 1000.0 + epoch * 100,
            'cache_hit_rate': 0.7 + epoch * 0.05,
        })
        
        test_logger.log_chess(epoch, {
            'elo_estimate': 1500.0 + epoch * 50,
            'win_rate': 0.4 + epoch * 0.05,
            'draw_rate': 0.3,
            'loss_rate': 0.3 - epoch * 0.05,
        })
        
        test_logger.log_hardware(epoch)
        
        time.sleep(0.5)
    
    time.sleep(2)  # Wait for async writes
    
    # Retrieve and print metrics
    print("\nLatest Training Metrics:")
    metrics = test_logger.get_latest_metrics('training_metrics', limit=3)
    for m in metrics:
        print(f"  Epoch {m['epoch']}: Loss={m['loss_total']:.3f}, Acc={m['accuracy_top1']:.3f}")
    
    test_logger.close()
    print("\nTest completed successfully!")
