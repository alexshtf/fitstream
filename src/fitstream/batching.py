from typing import Iterable, Sequence

import torch


def iter_batches(
    *tensors, batch_size: int = 1, shuffle: bool = True, generator: torch.Generator | None = None
) -> Iterable[Sequence[torch.Tensor]]:
    """Yields batches from tensors, optionally shuffled.

    Args:
        *tensors: One or more tensors that share the same first dimension (sample axis). Each
            yielded batch contains slices from each tensor aligned on that axis.
        batch_size: Number of samples per batch. The final batch may be smaller if the sample
            count is not divisible by the batch size.
        shuffle: Whether to shuffle samples before batching. Shuffling uses the device of the
            first tensor.
        generator: Optional torch.Generator for deterministic shuffling.

    Yields:
        Tuples of tensors, one per input tensor, representing a batch.

    Notes:
        This function assumes all tensors have the same number of samples along dimension 0
        and live on the same device. It does not perform explicit validation.
    """
    if not tensors:
        return
    if not shuffle:
        tensor_batches = [tensor.split(batch_size) for tensor in tensors]
        yield from zip(*tensor_batches)
    else:
        device = tensors[0].device
        n_samples = tensors[0].shape[0]
        idx = torch.randperm(n_samples, device=device, generator=generator)
        for idx_chunk in idx.split(batch_size):
            yield tuple(x[idx_chunk] for x in tensors)
