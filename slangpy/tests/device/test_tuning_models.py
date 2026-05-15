# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from tuning import ExhaustiveSearch, RandomSearch
from tuning.config import IntTunableParam, TunableParam, TuningSpace


def make_space() -> TuningSpace:
    return TuningSpace(
        [
            TunableParam("Op", "IOp", ["A", "B", "C"]),
            IntTunableParam("Width", [1, 2, 4]),
        ]
    )


def collect_proposals(model) -> list[str]:
    model.initialize(make_space())
    proposals = []
    while not model.is_complete():
        proposals.append(str(model.propose()))
    return proposals


def test_random_search_is_seed_deterministic():
    assert collect_proposals(RandomSearch(budget=5, seed=123)) == collect_proposals(
        RandomSearch(budget=5, seed=123)
    )


def test_random_search_uses_seed_for_order():
    assert collect_proposals(RandomSearch(budget=5, seed=1)) != collect_proposals(
        RandomSearch(budget=5, seed=2)
    )


def test_random_search_respects_budget_without_duplicates():
    proposals = collect_proposals(RandomSearch(budget=4, seed=0))

    assert len(proposals) == 4
    assert len(set(proposals)) == 4


def test_random_search_none_budget_covers_whole_space():
    proposals = collect_proposals(RandomSearch(seed=0))

    assert len(proposals) == make_space().size()
    assert set(proposals) == set(collect_proposals(ExhaustiveSearch()))


def test_random_search_zero_budget_is_immediately_complete():
    model = RandomSearch(budget=0)
    model.initialize(make_space())

    assert model.is_complete()
    with pytest.raises(StopIteration):
        model.propose()


def test_random_search_rejects_negative_budget():
    with pytest.raises(ValueError):
        RandomSearch(budget=-1)


def test_random_search_best_uses_reported_elapsed_time():
    model = RandomSearch(budget=3, seed=0)
    model.initialize(make_space())
    configs = [model.propose(), model.propose(), model.propose()]

    model.report(configs[0], 3.0)
    model.report(configs[1], 1.0)
    model.report(configs[2], 2.0)

    assert model.best() == configs[1]
    assert [r.iteration for r in model.history] == [0, 1, 2]
