import random

from scripts.make_grouped_embedding_split import exact_group_subset


def test_exact_group_subset_is_exact_and_deterministic():
    groups = [("a", 5), ("b", 4), ("c", 3), ("d", 2), ("e", 1)]

    first = exact_group_subset(groups, target=7, rng=random.Random(42))
    second = exact_group_subset(groups, target=7, rng=random.Random(42))

    sizes = dict(groups)
    assert first == second
    assert sum(sizes[group] for group in first) == 7
