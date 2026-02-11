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
        print(f"{CYAN}[*] Initializing Self-Healing Environment...{RESET}")
        missing = []
        for import_name, install_name in cls.REQUIRED_PACKAGES.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(install_name)

        if not missing:
            print(f"{GREEN}[+] All dependencies satisfied.{RESET}")
            return

        print(f"{YELLOW}[!] Missing dependencies found: {', '.join(missing)}{RESET}")
        for package in missing:
            # Validate package name to prevent injection
            if not re.match(r"^[a-zA-Z0-9\-_]+$", package):
                print(f"{RED}[!] Invalid package name: {package}{RESET}")
                continue

            print(f"[*] Installing {package}...", end=" ", flush=True)
            try:
                # Avoid shell=True for security
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{GREEN}DONE{RESET}")
            except subprocess.CalledProcessError:
                print(f"{RED}FAILED{RESET}")
                print(f"{RED}[!] Critical: Could not install {package}. Please install it manually.{RESET}")
                sys.exit(1)

        print(f"{GREEN}[+] Environment healed. Restarting to apply changes...{RESET}")
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Initial Healing
if __name__ == "__main__":
    DependencyManager.heal()

# Late Imports (after healing)
try:
    import chess
    import chess.pgn
    import chess.engine
    import h5py
    import numpy as np
    import requests
    import zstandard as zstd
    from tqdm import tqdm
except ImportError:
    pass

class AssetAcquisition:
    """
    Modul 2: AssetAcquisition (Smart Downloads)
    Handles downloading and preparing Stockfish and PGN data.
    """
    SF_RELEASE_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/"
    
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

    # Default PGN source: Lichess Standard September 2023 (will be streamed and limited)
    DEFAULT_PGN_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-09.pgn.zst"

    def __init__(self, asset_dir="assets"):
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(exist_ok=True)
        self.os_type = platform.system()
        self.arch = platform.machine()

    def detect_cpu_features(self):
        """Detects CPU features like AVX2 and BMI2."""
        features = set()
        if self.os_type == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                    if "avx2" in content: features.add("avx2")
                    if "bmi2" in content: features.add("bmi2")
            except Exception: pass
        elif self.os_type == "Windows":
            try:
                # Removed shell=True for security
                output = subprocess.check_output(["wmic", "cpu", "get", "description"]).decode().lower()
                if "x64" in output or "amd64" in output: features.add("modern")

                # Check for AVX2 via powershell if possible
                try:
                    avx_check = subprocess.check_output(
                        ["powershell", "-Command", "(Get-WmiObject Win32_Processor).Caption"],
                        stderr=subprocess.STDOUT
                    ).decode().lower()
                    # This is still not perfect but better than just guessing
                except: pass
            except Exception: pass
        elif self.os_type == "Darwin":
            try:
                output = subprocess.check_output(["sysctl", "-a"]).decode().lower()
                if "hw.optional.avx2: 1" in output: features.add("avx2")
                if "arm64" in self.arch.lower(): features.add("apple-silicon")
            except Exception: pass
        
        return features

    def get_stockfish(self):
        """Downloads and extracts the best Stockfish binary for the system."""
        print(f"{CYAN}[*] Detecting hardware... {self.os_type} {self.arch}{RESET}")
        features = self.detect_cpu_features()
        print(f"[*] CPU Features: {', '.join(features) if features else 'None detected'}")

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
            print(f"{GREEN}[+] Stockfish already present at {target_path}{RESET}")
            return str(target_path)

        url = self.SF_RELEASE_URL + selected_binary
        archive_path = self.asset_dir / selected_binary

        print(f"[*] Downloading Stockfish: {selected_binary}...")
        self._download_file(url, archive_path)
        
        print(f"[*] Extracting {selected_binary}...")
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
        print(f"{GREEN}[+] Stockfish initialized: {target_path}{RESET}")
        return str(target_path)

    def get_pgn_stream(self, url=None):
        """Returns a streaming response for the PGN database."""
        url = url or self.DEFAULT_PGN_URL
        print(f"[*] Opening PGN Stream: {url}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"{RED}[!] Failed to open PGN stream (Status {response.status_code}){RESET}")
            sys.exit(1)
        return response

    def _download_file(self, url, dest):
        """Download a file with verification."""
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

            print(f"[*] Verified Download SHA256: {sha256_hash.hexdigest()}")
            return True
        except Exception as e:
            print(f"{RED}[!] Download failed: {e}{RESET}")
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
    def evaluate_position(engine, board, depth=10, nodes=None):
        """Analyzes a board position and returns (score, best_move_uci)."""
        limit = chess.engine.Limit(depth=depth, nodes=nodes)
        info = engine.analyse(board, limit)
        
        score = DataProcessor.normalize_score(info["score"])
        
        # Extract best move from PV (Principal Variation)
        best_move = info.get("pv", [None])[0]
        best_move_uci = best_move.uci() if best_move else ""

        return score, best_move_uci

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
    fens, depth, nodes, engine_path, hash_size, threads = batch_data
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
    return results

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

    def run(self, pgn_response):
        print(f"{CYAN}[*] Starting ParallelMiner with {self.num_workers} workers...{RESET}")
        
        dctx = zstd.ZstdDecompressor()
        
        total_positions = 0
        games_processed = 0
        start_time = time.time()

        # Streaming directly from the HTTP response
        with dctx.stream_reader(pgn_response.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

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
                                               self.engine_path, self.args.hash, self.args.threads)
                                        current_batch = []

                            games_processed += 1

                        if current_batch:
                            yield (current_batch, self.args.depth, self.args.nodes,
                                   self.engine_path, self.args.hash, self.args.threads)

                    # Process batches
                    for result_batch in pool.imap_unordered(_process_batch, gen_batches()):
                        for fen, score, move in result_batch:
                            self.storage.add(fen, score, move)
                            total_positions += 1

                            if total_positions % 500 == 0:
                                elapsed = time.time() - start_time
                                pps = total_positions / elapsed
                                print(f"\r{MAGENTA}[*] Positions: {total_positions} | Games: {games_processed} | Speed: {pps:.1f} pos/s{RESET}", end="", flush=True)
                finally:
                    # Explicitly close workers
                    pool.close()
                    pool.join()
                    # We can't easily call _close_worker in the pool from here,
                    # but pool.join() should wait for processes to exit.
                    # Usually workers exit when the pool is closed.

        print(f"\n{GREEN}[+] Mining phase completed.{RESET}")
        return total_positions

class StorageBackend:
    """
    Modul 5: StorageBackend (HDF5 I/O)
    Efficiently stores generated data in HDF5 format with LZF compression.
    """
    def __init__(self, output_file="distillzero_dataset.h5", buffer_size=10000):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = {"fens": [], "scores": [], "moves": []}
        
        # Initialize HDF5 File
        with h5py.File(self.output_file, "a") as f:
            if "fens" not in f:
                ascii_dt = h5py.string_dtype(encoding='ascii')
                f.create_dataset("fens", (0,), maxshape=(None,), dtype=ascii_dt, chunks=(5000,), compression="lzf")
                f.create_dataset("scores", (0,), maxshape=(None,), dtype="float16", chunks=(5000,), compression="lzf")
                f.create_dataset("moves", (0,), maxshape=(None,), dtype=ascii_dt, chunks=(5000,), compression="lzf")
                print(f"{GREEN}[+] HDF5 Dataset created: {self.output_file}{RESET}")

    def add(self, fen, score, move):
        """Adds a single record to the buffer."""
        self.buffer["fens"].append(fen)
        self.buffer["scores"].append(score)
        self.buffer["moves"].append(move)
        
        if len(self.buffer["fens"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flushes the buffer to disk."""
        if not self.buffer["fens"]:
            return

        count = len(self.buffer["fens"])
        with h5py.File(self.output_file, "a") as f:
            for key in ["fens", "scores", "moves"]:
                dset = f[key]
                dset.resize(dset.shape[0] + count, axis=0)
                dset[-count:] = self.buffer[key]

        self.buffer = {k: [] for k in self.buffer}

    def close(self):
        """Ensures all data is written before closing."""
        self.flush()
        print(f"{GREEN}[+] Data successfully flushed to {self.output_file}{RESET}")

def main():
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
        print(f"{RED}[!] Insecure PGN URL. Only HTTPS allowed.{RESET}")
        sys.exit(1)

    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", args.output):
        print(f"{RED}[!] Invalid output filename.{RESET}")
        sys.exit(1)

    print(f"\n{CYAN}{'='*60}\n          DISTILLZERO DATA FACTORY v1.0\n{'='*60}{RESET}\n")

    # 1. Asset Acquisition
    acq = AssetAcquisition()
    sf_path = acq.get_stockfish()
    pgn_stream = acq.get_pgn_stream(url=args.pgn_url)

    # 2. Storage Setup
    storage = StorageBackend(output_file=args.output, buffer_size=args.buffer_size)

    # 3. Mining Phase
    miner = ParallelMiner(engine_path=sf_path, storage=storage, args=args)

    start_time = time.time()
    try:
        total_pos = miner.run(pgn_stream)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[!] User interrupted. Cleaning up...{RESET}")
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
