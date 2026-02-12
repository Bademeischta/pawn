import torch

def safe_load_checkpoint(path, device):
    """Load checkpoint with security and backwards compatibility."""
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


def safe_save(obj, path):
    """
    Save an object to a path safely using a temporary file.
    Prevents corruption if the process is interrupted during write.
    """
    from pathlib import Path
    import os

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save to temporary file
    import torch
    torch.save(obj, str(tmp_path))

    # Atomic rename (on POSIX)
    os.replace(tmp_path, path)
