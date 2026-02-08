import sys
import os
import platform
import subprocess
import time
import logging
import shutil
import stat
import json
import multiprocessing
import traceback
import io
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format=f"{CYAN}[%(asctime)s]{RESET} %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DistillZero")

class DependencyManager:
    """
    Modul 1: Self-Healing Environment.
    Checks and installs required packages automatically.
    """
    REQUIRED_PACKAGES = {
        "chess": "python-chess",
        "h5py": "h5py",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "requests": "requests",
        "zstandard": "zstandard"
    }

    @classmethod
    def check_and_install(cls):
        logger.info("Checking environment dependencies...")
        missing = []
        
        for import_name, install_name in cls.REQUIRED_PACKAGES.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(install_name)
        
        if missing:
            logger.warning(f"{YELLOW}Missing packages detected: {', '.join(missing)}{RESET}")
            for pkg in missing:
                print(f"Installing {pkg}...", end=" ")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
                    print(f"{GREEN}OK{RESET}")
                except subprocess.CalledProcessError:
                    print(f"{RED}FAILED{RESET}")
                    logger.critical(f"Could not install {pkg}. Please install manually.")
                    sys.exit(1)
            logger.info(f"{GREEN}Environment healed. Restarting script to load new modules...{RESET}")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            logger.info(f"{GREEN}All dependencies satisfied.{RESET}")

# Ensure dependencies before importing them
DependencyManager.check_and_install()

# Late imports after dependency check
import chess
import chess.pgn
import chess.engine
import h5py
import numpy as np
import requests
import zstandard as zstd
from tqdm import tqdm

class AssetAcquisition:
    """
    Modul 2: Smart Downloads.
    Handles Stockfish binary and PGN database acquisition.
    """
    STOCKFISH_VERSION = "sf_16.1"
    
    # Official GitHub Release Links (Snapshot)
    STOCKFISH_URLS = {
        "Windows": {
            "AMD64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-windows-x86-64-avx2.zip",
        },
        "Linux": {
            "x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar",
        },
        "Darwin": {
            "x86_64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-macos-x86-64-avx2.tar",
            "arm64": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-macos-m1-apple-silicon.tar"
        }
    }
    
    # Small reliable Lichess dump (Jan 2013) for factory testing/production start
    PGN_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"

    def __init__(self, work_dir="assets"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.stockfish_path = None
        self.pgn_path = self.work_dir / "data.pgn.zst"

    def get_stockfish(self):
        system = platform.system()
        machine = platform.machine()
        
        # Normalize machine names
        if machine == "AMD64": machine = "x86_64"
        
        logger.info(f"Detected System: {system} {machine}")
        
        expected_bin_name = "stockfish.exe" if system == "Windows" else "stockfish"
        final_path = self.work_dir / expected_bin_name
        
        if final_path.exists():
            logger.info(f"{GREEN}Stockfish binary found at {final_path}{RESET}")
            self.stockfish_path = str(final_path)
            return self.stockfish_path

        # Determine URL
        try:
            url = self.STOCKFISH_URLS[system][machine]
        except KeyError:
            logger.error(f"{RED}No pre-defined Stockfish URL for {system} {machine}. Please download manually.{RESET}")
            sys.exit(1)

        logger.info(f"Downloading Stockfish from {url}...")
        archive_path = self.work_dir / "stockfish_archive"
        self._download_file(url, archive_path)
        
        logger.info("Extracting Stockfish...")
        if url.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as z:
                # Find the binary in the zip
                for name in z.namelist():
                    if name.endswith(".exe") or "stockfish" in name:
                        if "/" in name and not name.endswith("/"): # It's a file in a subdir
                            source = z.open(name)
                            target = open(final_path, "wb")
                            with source, target:
                                shutil.copyfileobj(source, target)
                            break
        else: # tar
            import tarfile
            with tarfile.open(archive_path, 'r') as t:
                for member in t.getmembers():
                    if "stockfish" in member.name and member.isfile():
                        f = t.extractfile(member)
                        with open(final_path, "wb") as out:
                            shutil.copyfileobj(f, out)
                        break
        
        # Cleanup
        if archive_path.exists(): archive_path.unlink()
        
        # Make executable
        if system != "Windows":
            st = os.stat(final_path)
            os.chmod(final_path, st.st_mode | stat.S_IEXEC)
            
        if not final_path.exists():
            logger.error("Failed to extract Stockfish binary.")
            sys.exit(1)
            
        self.stockfish_path = str(final_path)
        logger.info(f"{GREEN}Stockfish ready.{RESET}")
        return self.stockfish_path

    def get_pgn_stream(self):
        """Returns a generator yielding lines from the PGN."""
        if not self.pgn_path.exists():
            logger.info(f"Downloading PGN Database from {self.PGN_URL}...")
            self._download_file(self.PGN_URL, self.pgn_path)
        
        logger.info(f"Opening PGN stream from {self.pgn_path}")
        return self.pgn_path

    def _download_file(self, url, dest_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            colour='green'
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

class DataProcessor:
    """
    Modul 3: The Core Logic.
    Handles position evaluation and score normalization.
    """
    @staticmethod
    def normalize_score(score: chess.engine.PovScore) -> float:
        """
        Converts chess engine score to -1..1 range.
        CP: tanh(cp/400)
        Mate: +1 or -1
        """
        mate = score.white().mate()
        if mate is not None:
            return 1.0 if mate > 0 else -1.0
        
        cp = score.white().score()
        if cp is None:
            return 0.0 # Should not happen usually
            
        return np.tanh(cp / 400.0)

class StorageBackend:
    """
    Modul 5: HDF5 I/O.
    Efficient storage with buffering and chunking.
    """
    def __init__(self, filename="distillzero_dataset.h5", buffer_size=10000):
        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = {
            "fens": [],
            "scores": [],
            "moves": []
        }
        
        # Initialize file and datasets
        with h5py.File(self.filename, 'a') as f:
            if "fens" not in f:
                # Variable length string dtype
                dt_str = h5py.string_dtype(encoding='ascii')
                
                f.create_dataset("fens", (0,), maxshape=(None,), dtype=dt_str, chunks=(5000,), compression="gzip")
                f.create_dataset("scores", (0,), maxshape=(None,), dtype="float16", chunks=(5000,), compression="gzip")
                f.create_dataset("moves", (0,), maxshape=(None,), dtype=dt_str, chunks=(5000,), compression="gzip")
    
    def add(self, fen, score, move):
        self.buffer["fens"].append(fen)
        self.buffer["scores"].append(score)
        self.buffer["moves"].append(move)
        
        if len(self.buffer["fens"]) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        if not self.buffer["fens"]:
            return
            
        count = len(self.buffer["fens"])
        with h5py.File(self.filename, 'a') as f:
            for key in ["fens", "scores", "moves"]:
                dset = f[key]
                dset.resize(dset.shape[0] + count, axis=0)
                dset[-count:] = self.buffer[key]
                
        self.buffer = {k: [] for k in self.buffer}
        # logger.info(f"Flushed {count} records to HDF5.") # Keep console clean

    def close(self):
        self.flush()

# Global worker initializer and function for multiprocessing
worker_engine_path = None
worker_engine = None

def worker_init(engine_path):
    global worker_engine_path, worker_engine
    worker_engine_path = engine_path
    try:
        # Check if Windows needs a specific creation flag to avoid popping up windows
        if platform.system() == 'Windows':
            worker_engine = chess.engine.SimpleEngine.popen_uci(engine_path, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            worker_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            
        worker_engine.configure({"Hash": 16, "Threads": 1})
    except Exception as e:
        # If initialization fails, the worker will fail tasks, handled by robust try/catch
        pass

def worker_process_batch(batch_fens):
    """
    Processes a batch of FENs.
    Returns list of (fen, score_val, best_move_uci)
    """
    global worker_engine
    results = []
    
    # Robustness: Re-init engine if crashed or not existing
    if worker_engine is None or worker_engine.transport.get_return_code() is not None:
        try:
            if platform.system() == 'Windows':
                worker_engine = chess.engine.SimpleEngine.popen_uci(worker_engine_path, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                worker_engine = chess.engine.SimpleEngine.popen_uci(worker_engine_path)
            worker_engine.configure({"Hash": 16, "Threads": 1})
        except Exception:
            return [] # Fail silently this batch

    limit = chess.engine.Limit(depth=10) # Speed/Quality balance
    
    for fen in batch_fens:
        try:
            board = chess.Board(fen)
            # Analyze
            info = worker_engine.analyse(board, limit)
            score = info["score"]
            
            # Normalize
            score_val = DataProcessor.normalize_score(score)
            
            # Best Move (for policy head) - simplistic, taking the PV[0] if available
            best_move = info.get("pv", [None])[0]
            best_move_uci = best_move.uci() if best_move else ""
            
            results.append((fen, score_val, best_move_uci))
        except chess.engine.EngineTerminatedError:
            # Engine crashed, abandon batch, worker will re-init next time
            worker_engine.quit()
            worker_engine = None
            break
        except Exception:
            continue
            
    return results

class ParallelMiner:
    """
    Modul 4: High-Performance Computing.
    Manages PGN streaming and parallel processing.
    """
    def __init__(self, engine_path, storage):
        self.engine_path = engine_path
        self.storage = storage
        self.num_workers = max(1, multiprocessing.cpu_count() - 2)
        
    def run(self, pgn_path):
        logger.info(f"Starting mining with {self.num_workers} workers...")
        
        # Open Compressed Stream
        dctx = zstd.ZstdDecompressor()
        
        batch_size = 100
        current_batch = []
        
        start_time = time.time()
        total_positions = 0
        
        with open(pgn_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                # Wrap the zstd stream with TextIOWrapper for line-based reading
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                # Setup Pool
                with multiprocessing.Pool(processes=self.num_workers, initializer=worker_init, initargs=(self.engine_path,)) as pool:
                    
                    # Generator to read games and yield batches of FENs
                    def batch_generator():
                        batch = []
                        while True:
                            try:
                                game = chess.pgn.read_game(text_stream)
                            except Exception:
                                break # EOF or error
                                
                            if game is None:
                                break
                                
                            board = game.board()
                            move_count = 0
                            for move in game.mainline_moves():
                                board.push(move)
                                move_count += 1
                                # Skip opening (first 10 moves)
                                if move_count > 20: 
                                    batch.append(board.fen())
                                    if len(batch) >= batch_size:
                                        yield batch
                                        batch = []
                        if batch:
                            yield batch

                    # Map batches to workers
                    # imap_unordered for better throughput
                    for result_batch in pool.imap_unordered(worker_process_batch, batch_generator()):
                        if result_batch:
                            for fen, score, move in result_batch:
                                self.storage.add(fen, score, move)
                                total_positions += 1
                                
                                if total_positions % 1000 == 0:
                                    elapsed = time.time() - start_time
                                    rate = total_positions / elapsed
                                    print(f"\rPositions: {total_positions} | Rate: {rate:.1f} pos/s", end="")

        self.storage.close()
        print()
        logger.info(f"{GREEN}Mining Complete.{RESET}")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    
    print(f"{CYAN}=== DISTILLZERO DATA FACTORY ==={RESET}")
    
    # 1. Setup Assets
    assets = AssetAcquisition()
    engine_path = assets.get_stockfish()
    pgn_path = assets.get_pgn_stream()
    
    # 2. Setup Storage
    storage = StorageBackend()
    
    # 3. Run Pipeline
    miner = ParallelMiner(engine_path, storage)
    
    try:
        start_t = time.time()
        miner.run(pgn_path)
        duration = time.time() - start_t
        
        # Summary
        with h5py.File("distillzero_dataset.h5", "r") as f:
            count = len(f["fens"])
            
        print(f"\n{GREEN}SUCCESS REPORT:{RESET}")
        print(f"Total Time: {duration:.2f}s")
        print(f"Total Positions: {count}")
        print(f"Output File: distillzero_dataset.h5")
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Flushing data...")
        storage.close()
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal Error: {e}")
        traceback.print_exc()
        sys.exit(1)
