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
