import torch
import logging
import logging.config
import os
from pathlib import Path
from typing import Union, Any

def setup_logging(level=logging.INFO):
    """Set up centralized logging configuration."""
    LOG_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'distillzero.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'detailed',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
            },
        }
    }
    logging.config.dictConfig(LOG_CONFIG)


def safe_load_checkpoint(path: Union[str, Path], device: torch.device):
    """Load checkpoint with security and backwards compatibility."""
    path = str(path)
    try:
        # PyTorch 2.4+: Secure loading
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch < 2.4: Fall back (with warning)
        import warnings
        warnings.warn(
            "Using torch.load without weights_only=True. "
            "Upgrade to PyTorch 2.4+ for secure loading.",
            UserWarning
        )
        return torch.load(path, map_location=device)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "WeightsUnpickler error" in msg or "Unsupported global" in msg:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Secure weights-only load failed for %s; retrying with weights_only=False. "
                "Only do this for checkpoints from a trusted source.",
                path
            )
            try:
                return torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=device)
        raise


def safe_save(obj: Any, path: Union[str, Path]):
    """
    Save an object to a path safely using a temporary file.
    Prevents corruption if the process is interrupted during write.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save to temporary file
    torch.save(obj, str(tmp_path))

    # Atomic rename (Atomic on POSIX, atomic on Windows 3.3+ if target exists)
    try:
        os.replace(tmp_path, path)
    except OSError:
        # Fallback for older Windows or file lock issues
        if path.exists():
            path.unlink()
        os.rename(tmp_path, path)
