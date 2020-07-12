import pytest
from data_util.dataset_utils import flatten_collate


def test_flatten_collate_should_remove_null():
    in_batch = [None, "a", "b", None, "c"]
    expected_output = ["a", "b", "c"]
    actual_output = flatten_collate(in_batch)
    assert actual_output == expected_output


def test_flatten_list():
    in_batch = [["s11", "s12", "s13"], ["s21", "s22", "s23"], ["s31", "s32", "s33"]]
    out_batch = ["s11", "s12", "s13", "s21", "s22", "s23", "s31", "s32", "s33"]
    actual_output = flatten_collate(in_batch)
    assert actual_output == out_batch