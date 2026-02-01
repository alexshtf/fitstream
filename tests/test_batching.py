import torch

from fitstream.batching import iter_batches


def test_iter_batches_no_shuffle_preserves_order() -> None:
    feature_tensor = torch.arange(12).view(6, 2)
    target_tensor = torch.arange(6)

    batch_tuples = list(iter_batches(feature_tensor, target_tensor, batch_size=4, shuffle=False))

    assert len(batch_tuples) == 2
    first_features, first_targets = batch_tuples[0]
    second_features, second_targets = batch_tuples[1]

    assert torch.equal(first_features, feature_tensor[:4])
    assert torch.equal(first_targets, target_tensor[:4])
    assert torch.equal(second_features, feature_tensor[4:])
    assert torch.equal(second_targets, target_tensor[4:])


def test_iter_batches_shuffle_is_deterministic_with_generator() -> None:
    feature_tensor = torch.arange(10).view(10, 1)
    target_tensor = torch.arange(10) * 10

    random_generator = torch.Generator().manual_seed(1234)
    batch_tuples = list(iter_batches(feature_tensor, target_tensor, batch_size=3, shuffle=True, generator=random_generator))

    expected_indices = torch.randperm(10, generator=torch.Generator().manual_seed(1234))
    expected_features = feature_tensor[expected_indices]
    expected_targets = target_tensor[expected_indices]

    observed_features = torch.cat([batch_tuple[0] for batch_tuple in batch_tuples], dim=0)
    observed_targets = torch.cat([batch_tuple[1] for batch_tuple in batch_tuples], dim=0)

    assert torch.equal(observed_features, expected_features)
    assert torch.equal(observed_targets, expected_targets)
