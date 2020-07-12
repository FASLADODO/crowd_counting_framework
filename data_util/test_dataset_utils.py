import pytest
from data_util.dataset_utils import _flatten_collate


# def test_flatten_collate_should_remove_null():
#     in_batch = [None, "a", "b", None, "c"]
#     expected_output = ["a", "b", "c"]
#     actual_output = _flatten_collate(in_batch)
#     assert actual_output == expected_output


def test_flatten_list():
    in_batch = [(["d11", "d12", "d13"],["l11", "l12", "l13"]),(["d21", "d22", "d23"],["l21", "l22", "l23"]), (["d31", "d32", "d33"],["l31", "l32", "l33"])]
    out_batch = [("d11", "l11"), ("d12", "l12"), ("d13", "l13"), ("d21", "l21"), ("d22", "l22"), ("d23", "l23"), ("d31", "l31"), ("d32", "l32"), ("d33", "l33")]
    actual_output = _flatten_collate(in_batch)
    assert actual_output == out_batch
    for data in actual_output:
        print(len(data))