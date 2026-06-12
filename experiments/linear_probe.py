"""Evaluate representation quality at each PC level via a linear probe."""


def extract_representations(model, dataloader, level: int, device):
    """Extract frozen representations at a given PC level."""
    raise NotImplementedError


def train_linear_probe(representations, labels, n_epochs=20):
    raise NotImplementedError


def evaluate_probe(probe, representations, labels):
    raise NotImplementedError


def run(model, dataloader_train, dataloader_test, n_levels: int, device):
    """Train and evaluate a linear probe at every PC level."""
    raise NotImplementedError
