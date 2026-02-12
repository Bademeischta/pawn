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
import argparse
import re
from pathlib import Path
from datetime import datetime
from utils import setup_logging

logger = logging.getLogger(__name__)

# --- ANSI Escape Codes for Colored Output ---
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

class DependencyManager:
    """
    Modul 1: DependencyManager (Self-Healing Environment)
    Checks for required libraries and installs them if missing.
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
    def heal(cls):
        """Checks and installs missing packages."""
        logger.info("Initializing Self-Healing Environment...")
        missing = []
        for import_name, install_name in cls.REQUIRED_PACKAGES.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(install_name)

        if not missing:
            logger.info("All dependencies satisfied.")
            return

        logger.warning(f"Missing dependencies found: {', '.join(missing)}")
        for package in missing:
            # Validate package name to prevent injection
            if not re.match(r"^[a-zA-Z0-9\-_]+$", package):
                logger.error(f"Invalid package name: {package}")
                continue

            logger.info(f"Installing {package}...")
            try:
                # Avoid shell=True for security
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                logger.critical(f"Could not install {package}. Please install it manually.")
                sys.exit(1)

        logger.info("Environment healed. Restarting to apply changes...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Late Imports (after healing)
try:
    import chess
    import chess.pgn
    import chess.engine
    import chess.polyglot
    import h5py
    import numpy as np
    import requests
    import zstandard as zstd
    from tqdm import tqdm
except ImportError:
    pass

class CPUDetector:
    """Robust CPU feature detection across platforms."""
    @staticmethod
    def get_features():
        features = set()
        os_type = platform.system()
        try:
            if os_type == "Linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        content = f.read().lower()
                        if "avx2" in content: features.add("avx2")
                        if "bmi2" in content: features.add("bmi2")
                except FileNotFoundError:
                    logger.debug("/proc/cpuinfo not found")
            elif os_type == "Windows":
                # Use more robust detection on Windows
                try:
                    output = subprocess.check_output(["wmic", "cpu", "get", "description"]).decode().lower()
                    if "x64" in output or "amd64" in output: features.add("modern")
                except Exception as e:
                    logger.debug(f"Windows CPU detection via wmic failed: {e}")
            elif os_type == "Darwin":
                try:
                    output = subprocess.check_output(["sysctl", "-a"]).decode().lower()
                    if "hw.optional.avx2: 1" in output: features.add("avx2")
                    if "arm64" in platform.machine().lower(): features.add("apple-silicon")
                except: pass
        except Exception as e:
            logging.warning(f"CPU detection failed: {e}")
        return features

class AssetAcquisition:
    """
    Modul 2: AssetAcquisition (Smart Downloads)
    Handles downloading and preparing Stockfish and PGN data.
    """
    SF_VERSION = "16.1"
    SF_RELEASE_URL = f"https://github.com/official-stockfish/Stockfish/releases/download/sf_{SF_VERSION}/"
    
    # Mapping of features to filenames
    SF_BINARIES = {
        "Linux": {
            "avx2": "stockfish-ubuntu-x86-64-avx2.tar",
            "bmi2": "stockfish-ubuntu-x86-64-bmi2.tar",
            "modern": "stockfish-ubuntu-x86-64-modern.tar",
            "x86-64": "stockfish-ubuntu-x86-64.tar"
        },
        "Windows": {
            "avx2": "stockfish-windows-x86-64-avx2.zip",
            "bmi2": "stockfish-windows-x86-64-bmi2.zip",
            "modern": "stockfish-windows-x86-64-modern.zip",
            "x86-64": "stockfish-windows-x86-64.zip"
        },
        "Darwin": {
            "apple-silicon": "stockfish-macos-m1-apple-silicon.tar",
            "avx2": "stockfish-macos-x86-64-avx2.tar",
            "modern": "stockfish-macos-x86-64-modern.tar"
        }
    }

    # Hardcoded hashes for Stockfish 16.1 (example, should be verified)
    SF_HASHES = {
        "stockfish-ubuntu-x86-64-avx2.tar": "099a98c5643444458514167905187e1f409559560f4a86770f7f32997780005d",
        "stockfish-windows-x86-64-avx2.zip": "1f8f9037c8c6a677b102f5a60037f59798544a86770f7f32997780005d", # DUMMY
    }

    # Default PGN source: Lichess Standard September 2023 (will be streamed and limited)
    DEFAULT_PGN_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-09.pgn.zst"

    def __init__(self, asset_dir="assets"):
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(exist_ok=True)
        self.os_type = platform.system()
        self.arch = platform.machine()

    def get_stockfish(self):
        """Downloads and extracts the best Stockfish binary for the system."""
        logger.info(f"Detecting hardware... {self.os_type} {self.arch}")
        features = CPUDetector.get_features()
        logger.info(f"CPU Features: {', '.join(features) if features else 'None detected'}")

        # Priority selection
        os_binaries = self.SF_BINARIES.get(self.os_type, self.SF_BINARIES["Linux"])
        selected_binary = None
        
        if "apple-silicon" in features and "apple-silicon" in os_binaries:
            selected_binary = os_binaries["apple-silicon"]
        elif "avx2" in features and "avx2" in os_binaries:
            selected_binary = os_binaries["avx2"]
        elif "bmi2" in features and "bmi2" in os_binaries:
            selected_binary = os_binaries["bmi2"]
        elif "modern" in os_binaries:
            selected_binary = os_binaries["modern"]
        else:
            selected_binary = os_binaries["x86-64"]

        target_bin_name = "stockfish" + (".exe" if self.os_type == "Windows" else "")
        target_path = self.asset_dir / target_bin_name

        if target_path.exists():
            logger.info(f"Stockfish already present at {target_path}")
            return str(target_path)

        url = self.SF_RELEASE_URL + selected_binary
        archive_path = self.asset_dir / selected_binary

        expected_hash = self.SF_HASHES.get(selected_binary)
        logger.info(f"Downloading Stockfish: {selected_binary}...")
        self._download_file(url, archive_path, expected_hash=expected_hash)
        
        logger.info(f"Extracting {selected_binary}...")
        if selected_binary.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Find the largest file (likely the binary)
                members = [m for m in zip_ref.infolist() if not m.is_dir()]
                best_member = max(members, key=lambda m: m.file_size)
                data = zip_ref.read(best_member)
                with open(target_path, "wb") as f:
                    f.write(data)
        else:
            import tarfile
            with tarfile.open(archive_path, 'r') as tar_ref:
                # Find the largest file (likely the binary)
                members = [m for m in tar_ref.getmembers() if m.isfile()]
                best_member = max(members, key=lambda m: m.size)
                content = tar_ref.extractfile(best_member)
                with open(target_path, "wb") as f:
                    f.write(content.read())
        
        if not self.os_type == "Windows":
            target_path.chmod(target_path.stat().st_mode | stat.S_IEXEC)
        
        archive_path.unlink() # Cleanup
        logger.info(f"Stockfish initialized: {target_path}")
        return str(target_path)

    def get_pgn_stream(self, url=None):
        """Returns a streaming response for the PGN database."""
        url = url or self.DEFAULT_PGN_URL
        logger.info(f"Opening PGN Stream: {url}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            logger.error(f"Failed to open PGN stream (Status {response.status_code})")
            sys.exit(1)
        return response

    def _download_file(self, url, dest, expected_hash=None):
        """Download a file with SHA256 verification."""
        if not url.startswith("https://"):
            raise ValueError(f"Insecure URL: {url}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            import hashlib
            sha256_hash = hashlib.sha256()

            with open(dest, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=dest.name, colour='green'
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        sha256_hash.update(chunk)
                        pbar.update(len(chunk))

            computed_hash = sha256_hash.hexdigest()
            logger.info(f"Downloaded SHA256: {computed_hash}")

            if expected_hash and computed_hash != expected_hash:
                raise ValueError(f"Hash mismatch! Expected {expected_hash}, got {computed_hash}")

            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

class DataProcessor:
    """
    Modul 3: DataProcessor (The Core Logic)
    Handles position evaluation and score normalization.
    """
    @staticmethod
    def normalize_score(score: chess.engine.PovScore) -> float:
        """
        Normalizes engine score to [-1, 1] range.
        Formula: y = tanh(cp / 400.0)
        Mate handling: +1.0 for white wins, -1.0 for black wins.
        """
        cp = score.white().score()
        mate = score.white().mate()

        if mate is not None:
            return 1.0 if mate > 0 else -1.0
        
        if cp is not None:
            return float(np.tanh(cp / 400.0))

        return 0.0

    @staticmethod
    def evaluate_position(engine, board, depth=10, nodes=None, timeout=10.0):
        """Analyzes a board position and returns (score, best_move_uci)."""
        try:
            limit = chess.engine.Limit(depth=depth, nodes=nodes)
            info = engine.analyse(board, limit, timeout=timeout)

            score = DataProcessor.normalize_score(info["score"])

            # Extract best move from PV (Principal Variation)
            best_move = info.get("pv", [None])[0]
            best_move_uci = best_move.uci() if best_move else ""

            return score, best_move_uci
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError) as e:
            logging.error(f"Engine evaluation error: {e}")
            return 0.0, ""

# --- Multiprocessing Worker Logic ---
_worker_engine = None

def _init_worker(engine_path, hash_size, threads):
    global _worker_engine
    try:
        # Validate engine path
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        if platform.system() == "Windows":
            _worker_engine = chess.engine.SimpleEngine.popen_uci(engine_path, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            _worker_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        _worker_engine.configure({"Hash": hash_size, "Threads": threads})
    except Exception as e:
        print(f"{RED}[!] Worker Init Error: {e}{RESET}")
        _worker_engine = None

def _close_worker():
    global _worker_engine
    if _worker_engine:
        try:
            _worker_engine.quit()
        except:
            pass
        _worker_engine = None

def _process_batch(batch_data):
    """Worker task: processes a batch of FENs with robust engine restart."""
    global _worker_engine
    fens, depth, nodes, engine_path, hash_size, threads, batch_max_game_idx = batch_data
    results = []
    
    def ensure_engine():
        global _worker_engine
        if _worker_engine is None:
            _init_worker(engine_path, hash_size, threads)
        return _worker_engine is not None

    if not ensure_engine():
        return []

    for fen in fens:
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                continue

            # Analyze position type for balancing (basic heuristic)
            is_sharp = board.is_check() or any(board.is_capture(m) for m in board.legal_moves)

            score, best_move = DataProcessor.evaluate_position(_worker_engine, board, depth=depth, nodes=nodes)

            # Filter: Skip positions with Stockfish Eval between -0.3 and +0.3 (langweilige Remis)
            if -0.3 < score < 0.3:
                # Keep only 10% of "boring" draws
                if not is_sharp and np.random.random() > 0.1:
                    continue

            results.append((fen, score, best_move))
        except (chess.engine.EngineTerminatedError, Exception):
            # Attempt to restart engine once on error
            try:
                if _worker_engine: _worker_engine.quit()
            except Exception: pass
            _worker_engine = None
            if not ensure_engine(): break
            
            try:
                board = chess.Board(fen)
                score, best_move = DataProcessor.evaluate_position(_worker_engine, board, depth=depth, nodes=nodes)
                results.append((fen, score, best_move))
            except Exception:
                continue
    return results, batch_max_game_idx

class ParallelMiner:
    """
    Modul 4: ParallelMiner (High-Performance Computing)
    Orchestrates the data generation process using multiple processes.
    """
    def __init__(self, engine_path, storage, args):
        self.engine_path = engine_path
        self.storage = storage
        self.args = args
        self.num_workers = max(1, multiprocessing.cpu_count() - 2)

    def run(self, pgn_response, initial_games=0):
        logger.info(f"Starting ParallelMiner with {self.num_workers} workers...")
        logger.info(f"Resuming from game index: {initial_games}")
        
        dctx = zstd.ZstdDecompressor()
        
        total_positions = 0
        games_processed = initial_games
        start_time = time.time()

        # Streaming directly from the HTTP response
        with dctx.stream_reader(pgn_response.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            # Skip games if resuming
            if initial_games > 0:
                logger.info(f"Skipping {initial_games} games...")
                for _ in range(initial_games):
                    chess.pgn.read_game(text_stream)

            with multiprocessing.Pool(
                processes=self.num_workers,
                initializer=_init_worker,
                initargs=(self.engine_path, self.args.hash, self.args.threads)
            ) as pool:
                try:
                    def gen_batches():
                        nonlocal games_processed
                        current_batch = []
                        while True:
                            if self.args.max_games and games_processed >= self.args.max_games:
                                break

                            try:
                                game = chess.pgn.read_game(text_stream)
                            except Exception:
                                break
                                
                            if game is None: break

                            board = game.board()
                            # Filter by Rating (>= 2000 ELO)
                            white_elo = game.headers.get("WhiteElo", "0")
                            black_elo = game.headers.get("BlackElo", "0")
                            try:
                                if int(white_elo) < 2000 and int(black_elo) < 2000:
                                    continue
                            except ValueError:
                                continue

                            for i, move in enumerate(game.mainline_moves()):
                                board.push(move)
                                # Sample positions from moves 15-60 (Mittelspiel)
                                if 30 <= i <= 120: # 30 plies = move 15
                                    current_batch.append(board.fen())
                                    if len(current_batch) >= 100:
                                        yield (current_batch, self.args.depth, self.args.nodes,
                                               self.engine_path, self.args.hash, self.args.threads, games_processed)
                                        current_batch = []

                            games_processed += 1

                        if current_batch:
                            yield (current_batch, self.args.depth, self.args.nodes,
                                   self.engine_path, self.args.hash, self.args.threads, games_processed)

                    # Process batches with Zobrist deduplication
                    seen_hashes = set()
                    duplicates_removed = 0
                    accumulator = []
                    max_finished_game_idx = initial_games

                    for result_batch, batch_game_idx in pool.imap_unordered(_process_batch, gen_batches()):
                        max_finished_game_idx = max(max_finished_game_idx, batch_game_idx)
                        for fen, score, move in result_batch:
                            try:
                                # Standard AlphaZero/Stockfish deduplication via Zobrist hashing
                                board = chess.Board(fen)
                                zhash = chess.polyglot.zobrist_hash(board)

                                if zhash not in seen_hashes:
                                    seen_hashes.add(zhash)
                                    accumulator.append((fen, score, move))
                                    total_positions += 1

                                    # Memory-efficient batched flush to storage
                                    if len(accumulator) >= 1000:
                                        for f, s, m in accumulator:
                                            self.storage.add(f, s, m, games_processed=max_finished_game_idx)
                                        accumulator = []
                                else:
                                    duplicates_removed += 1
                            except Exception:
                                continue

                            if total_positions % 500 == 0:
                                elapsed = time.time() - start_time
                                pps = total_positions / elapsed
                                total_processed = total_positions + duplicates_removed
                                dedup_rate = (duplicates_removed / total_processed * 100) if total_processed > 0 else 0
                                print(f"\r{MAGENTA}[*] Positions: {total_positions} | Dedup: {dedup_rate:.1f}% | Speed: {pps:.1f} pos/s{RESET}", end="", flush=True)

                    # Final flush of deduplicated positions
                    for f, s, m in accumulator:
                        self.storage.add(f, s, m, games_processed=max_finished_game_idx)
                finally:
                    # Explicitly close workers
                    pool.close()
                    pool.join()
                    # We can't easily call _close_worker in the pool from here,
                    # but pool.join() should wait for processes to exit.
                    # Usually workers exit when the pool is closed.

        logger.info("Mining phase completed.")
        return total_positions, games_processed

class StorageBackend:
    """
    Modul 5: StorageBackend (HDF5 I/O)
    Efficiently stores generated data in HDF5 format with LZF compression.
    """
    def __init__(self, output_file="distillzero_dataset.h5", buffer_size=10000):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = {"fens": [], "scores": [], "moves": []}
        self.current_games_processed = 0
        
        # Initialize HDF5 File
        with h5py.File(self.output_file, "a") as f:
            if "fens" not in f:
                ascii_dt = h5py.string_dtype(encoding='ascii')
                f.create_dataset("fens", (0,), maxshape=(None,), dtype=ascii_dt, chunks=(5000,), compression="lzf")
                f.create_dataset("scores", (0,), maxshape=(None,), dtype="float16", chunks=(5000,), compression="lzf")
                f.create_dataset("moves", (0,), maxshape=(None,), dtype=ascii_dt, chunks=(5000,), compression="lzf")
                f.attrs["games_processed"] = 0
                logger.info(f"HDF5 Dataset created: {self.output_file}")
            else:
                self.current_games_processed = f.attrs.get("games_processed", 0)
                logger.info(f"Resuming existing dataset: {self.output_file} ({self.current_games_processed} games processed)")

    def get_games_processed(self):
        with h5py.File(self.output_file, "r") as f:
            return f.attrs.get("games_processed", 0)

    def set_games_processed(self, count):
        with h5py.File(self.output_file, "a") as f:
            f.attrs["games_processed"] = count

    def add(self, fen, score, move, games_processed=None):
        """Adds a single record to the buffer."""
        self.buffer["fens"].append(fen)
        self.buffer["scores"].append(score)
        self.buffer["moves"].append(move)
        if games_processed is not None:
            self.current_games_processed = games_processed
        
        if len(self.buffer["fens"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flushes the buffer to disk."""
        if not self.buffer["fens"] and self.current_games_processed == self.get_games_processed():
            return

        count = len(self.buffer["fens"])
        with h5py.File(self.output_file, "a") as f:
            if count > 0:
                for key in ["fens", "scores", "moves"]:
                    dset = f[key]
                    dset.resize(dset.shape[0] + count, axis=0)
                    dset[-count:] = self.buffer[key]

            f.attrs["games_processed"] = self.current_games_processed

        self.buffer = {k: [] for k in self.buffer}

    def close(self):
        """Ensures all data is written before closing."""
        self.flush()
        logger.info(f"Data successfully flushed to {self.output_file}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="DistillZero Data Factory - High-Performance Chess Data Generation")
    parser.add_argument("--depth", type=int, default=10, help="Stockfish evaluation depth (default: 10)")
    parser.add_argument("--nodes", type=int, default=None, help="Stockfish node limit (default: None)")
    parser.add_argument("--max-games", type=int, default=200000, help="Maximum number of games to process (default: 200,000)")
    parser.add_argument("--hash", type=int, default=16, help="Stockfish hash size in MB (default: 16)")
    parser.add_argument("--threads", type=int, default=1, help="Stockfish threads per worker (default: 1)")
    parser.add_argument("--output", type=str, default="distillzero_dataset.h5", help="Output HDF5 filename")
    parser.add_argument("--pgn-url", type=str, default=None, help="Custom Lichess PGN URL")
    parser.add_argument("--buffer-size", type=int, default=50000, help="HDF5 write buffer size (default: 50,000)")
    args = parser.parse_args()

    # Input validation
    if args.pgn_url and not args.pgn_url.startswith("https://"):
        logger.error("Insecure PGN URL. Only HTTPS allowed.")
        sys.exit(1)

    # Validate output path
    try:
        out_path = Path(args.output)
        # Ensure parent directory exists
        if not out_path.parent.exists():
            logger.info(f"Creating directory: {out_path.parent}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Invalid output path: {args.output} ({e})")
        sys.exit(1)

    logger.info("DISTILLZERO DATA FACTORY v1.0 Started")

    # 1. Asset Acquisition
    acq = AssetAcquisition()
    sf_path = acq.get_stockfish()
    pgn_stream = acq.get_pgn_stream(url=args.pgn_url)

    # 2. Storage Setup
    storage = StorageBackend(output_file=args.output, buffer_size=args.buffer_size)

    # 3. Mining Phase
    initial_games = storage.get_games_processed()
    miner = ParallelMiner(engine_path=sf_path, storage=storage, args=args)

    start_time = time.time()
    total_pos = 0
    try:
        total_pos, games_processed = miner.run(pgn_stream, initial_games=initial_games)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[!] User interrupted. Cleaning up...{RESET}")
    except Exception as e:
        logger.error(f"Error during mining: {e}")
        logger.error(traceback.format_exc())
    finally:
        storage.close()
        pgn_stream.close()

    total_time = time.time() - start_time

    # 4. Success Report
    print(f"\n{CYAN}{'='*60}\n               SUCCESS REPORT\n{'='*60}{RESET}")
    print(f"Total Time:     {total_time/3600:.2f} hours")
    print(f"Total Positions: {total_pos}")
    print(f"Average Speed:   {total_pos/total_time:.1f} pos/s")
    print(f"Output File:     {args.output}")
    print(f"{CYAN}{'='*60}{RESET}\n")

if __name__ == "__main__":
    # Windows fix for multiprocessing
    multiprocessing.freeze_support()

    # Self-healing first
    DependencyManager.heal()

    # Then run main
    main()
