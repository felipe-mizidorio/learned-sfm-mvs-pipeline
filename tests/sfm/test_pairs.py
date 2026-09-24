import pytest

from learned_sfm_mvs.sfm.pairs import (
    exhaustive_pairs,
    merge_pairs,
    read_pairs,
    sequential_pairs,
    write_pairs,
)

NAMES = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]


def test_sequential_pairs_overlap_2():
    assert sequential_pairs(NAMES, 2) == [
        ("a.jpg", "b.jpg"),
        ("a.jpg", "c.jpg"),
        ("b.jpg", "c.jpg"),
        ("b.jpg", "d.jpg"),
        ("c.jpg", "d.jpg"),
    ]


def test_sequential_pairs_overlap_larger_than_set_is_exhaustive():
    assert set(sequential_pairs(NAMES, 10)) == set(exhaustive_pairs(NAMES))


def test_sequential_pairs_rejects_non_positive_overlap():
    with pytest.raises(ValueError, match="overlap"):
        sequential_pairs(NAMES, 0)


def test_exhaustive_pairs_count():
    assert len(exhaustive_pairs(NAMES)) == 6


def test_merge_pairs_drops_reversed_duplicates_and_self_pairs():
    merged = merge_pairs(
        [("a.jpg", "b.jpg"), ("b.jpg", "c.jpg")],
        [("b.jpg", "a.jpg"), ("d.jpg", "d.jpg"), ("a.jpg", "d.jpg")],
    )
    assert merged == [("a.jpg", "b.jpg"), ("b.jpg", "c.jpg"), ("a.jpg", "d.jpg")]


def test_write_read_roundtrip(tmp_path):
    pairs = [("a.jpg", "sub/b.jpg"), ("c.jpg", "d.jpg")]
    path = tmp_path / "out" / "pairs.txt"
    write_pairs(pairs, path)
    assert path.read_text() == "a.jpg sub/b.jpg\nc.jpg d.jpg\n"
    assert read_pairs(path) == pairs


def test_write_pairs_rejects_whitespace_in_names(tmp_path):
    with pytest.raises(ValueError, match="whitespace"):
        write_pairs([("my frame.jpg", "b.jpg")], tmp_path / "pairs.txt")
