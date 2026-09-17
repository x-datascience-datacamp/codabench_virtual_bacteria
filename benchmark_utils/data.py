"""Small runtime helpers shared across solvers."""


def get_device():
    """CUDA when available, else CPU."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
