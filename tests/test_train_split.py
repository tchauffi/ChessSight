"""How a run is divided into train, validation and test.

The rule is shared between the data loaders and the mAP evaluator. It has to be:
when only the loaders knew it, evaluating a single-split dataset on ``val`` matched
no stored entry, reported ``map=-1`` every epoch, and left ``best`` pointing at
whichever epoch happened to run first.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from chesssight.train.dataset import SplitSpec, select_entries


@dataclass(frozen=True)
class Entry:
    """Just the two fields the split rule looks at."""

    id: str
    split: str


def synthetic_run(count: int = 200) -> list[Entry]:
    """A generated dataset: every sample carries the same stored split."""
    return [Entry(id=f"{index:06d}", split="train") for index in range(count)]


def annotated_run(count: int = 200) -> list[Entry]:
    """A real dataset that carries its own train/val/test division."""
    splits = ("train", "val", "test")
    return [Entry(id=f"{index:06d}", split=splits[index % 3]) for index in range(count)]


class TestSingleSplitDataset:
    def test_val_selects_real_samples_rather_than_nothing(self):
        # The regression: a synthetic run stores "train" for everything, so matching
        # the stored value literally leaves the val set empty and every metric
        # measured against it undefined.
        entries, source = select_entries(synthetic_run(), split="val")
        assert source == "hash"
        assert entries

    def test_the_three_splits_partition_the_dataset(self):
        everything = synthetic_run()
        parts = {
            name: {e.id for e in select_entries(everything, split=name)[0]}
            for name in ("train", "val", "test")
        }

        assert not parts["train"] & parts["val"]
        assert not parts["train"] & parts["test"]
        assert not parts["val"] & parts["test"]
        assert sum(len(part) for part in parts.values()) == len(everything)

    def test_the_val_fraction_is_honoured(self):
        entries, _ = select_entries(
            synthetic_run(2000), split="val", spec=SplitSpec(val_fraction=0.1)
        )
        assert 0.08 < len(entries) / 2000 < 0.12

    def test_the_split_is_stable_across_calls(self):
        first, _ = select_entries(synthetic_run(), split="val")
        second, _ = select_entries(synthetic_run(), split="val")
        assert [e.id for e in first] == [e.id for e in second]

    def test_a_different_fraction_moves_the_boundary(self):
        small, _ = select_entries(
            synthetic_run(2000), split="val", spec=SplitSpec(val_fraction=0.1)
        )
        large, _ = select_entries(
            synthetic_run(2000), split="val", spec=SplitSpec(val_fraction=0.3)
        )
        # Nested, not merely larger: growing the fraction must not reshuffle which
        # samples were already held out, or a resumed run would train on its own
        # validation set.
        assert {e.id for e in small} < {e.id for e in large}

    def test_the_test_fraction_is_honoured(self):
        entries, _ = select_entries(
            synthetic_run(2000),
            split="test",
            spec=SplitSpec(val_fraction=0.1, test_fraction=0.15),
        )
        assert 0.13 < len(entries) / 2000 < 0.17

    def test_adding_a_test_split_leaves_the_val_set_alone(self):
        # Test is carved off *above* val, so a number measured before this split
        # existed still refers to the same samples; what moves is training data.
        everything = synthetic_run(2000)
        without, _ = select_entries(
            everything, split="val", spec=SplitSpec(val_fraction=0.1, test_fraction=0.0)
        )
        with_test, _ = select_entries(
            everything, split="val", spec=SplitSpec(val_fraction=0.1, test_fraction=0.2)
        )
        assert [e.id for e in without] == [e.id for e in with_test]

    def test_asking_for_a_test_set_that_was_not_reserved_is_refused(self):
        # Silently returning the val set, or nothing at all, would both be read as
        # a held-out number.
        with pytest.raises(ValueError, match="no test set"):
            select_entries(
                synthetic_run(), split="test", spec=SplitSpec(test_fraction=0.0)
            )

    def test_a_spec_that_leaves_no_training_data_is_refused(self):
        with pytest.raises(ValueError, match="no training data"):
            SplitSpec(val_fraction=0.5, test_fraction=0.5)


class TestStoredSplitDataset:
    def test_a_dataset_carrying_its_own_division_is_respected(self):
        # ChessReD splits by *game* -- images from one game share a board, a room
        # and a camera -- so re-hashing it would leak all three across the boundary
        # and flatter every number measured against it.
        entries, source = select_entries(annotated_run(), split="val")
        assert source == "stored"
        assert {e.split for e in entries} == {"val"}

    def test_every_stored_split_is_reachable(self):
        for split in ("train", "val", "test"):
            entries, _ = select_entries(annotated_run(), split=split)
            assert {e.split for e in entries} == {split}

    def test_hashing_can_be_forced_on_a_stored_dataset(self):
        entries, source = select_entries(
            annotated_run(), split="val", split_source="hash"
        )
        assert source == "hash"
        assert {e.split for e in entries} != {"val"}


def test_all_returns_everything_either_way():
    for entries in (synthetic_run(), annotated_run()):
        selected, _ = select_entries(entries, split="all")
        assert len(selected) == len(entries)
