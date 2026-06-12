"""
Sequential task training on CIFAR-100 splits to measure catastrophic forgetting.
Compares PC composite (local weight updates) vs JEPA fine-tuned with backprop.
"""


def make_task_splits(dataset, n_tasks=10):
    """Partition CIFAR-100 into n_tasks disjoint class subsets."""
    raise NotImplementedError


def train_on_task(model, dataloader, task_id: int, device):
    raise NotImplementedError


def evaluate_all_tasks(model, task_dataloaders: list, device):
    """Return per-task accuracy after training on every seen task."""
    raise NotImplementedError


def run(model_pc, model_jepa, dataset_train, dataset_test, n_tasks=10, device="cpu"):
    """
    Train sequentially on all tasks, recording accuracy on all prior tasks
    after each new task. Returns forgetting curves for both models.
    """
    raise NotImplementedError
