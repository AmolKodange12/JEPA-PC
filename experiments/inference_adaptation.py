"""
Test whether iterative PC inference improves representations over a single
forward pass, especially under input corruption.
"""


def corrupt(images, corruption_type: str, severity: float):
    """Apply Gaussian noise, masking, or occlusion to a batch of images."""
    raise NotImplementedError


def evaluate_inference_adaptation(model, images, corruption_fn, n_steps=20):
    """
    1. Corrupt images.
    2. Single forward pass  → representation r_0.
    3. Run PC inference for n_steps → representation r_n.
    4. Compare linear-probe accuracy of r_0 vs r_n.
    """
    raise NotImplementedError


def run(model, dataloader, corruption_types: list[str], severities: list[float], device):
    raise NotImplementedError
