from aspect.collate import ColumnCollator


def test_column_collator():
    rows = [
        {
            "a": 1,
            "b": "x",
        },
        {
            "a": 2,
            "b": "y",
        },
    ]

    collator = ColumnCollator()
    batch = collator(rows)
    assert batch == {
        "a": [1, 2],
        "b": ["x", "y"],
    }


def test_column_collator_2d():
    rows = [
        {"x": [1., 2.], "y": 1.},
        {"x": [30, 4.], "y": 2.},
    ]
    collator = ColumnCollator()
    batch = collator(rows)
    assert batch == {
        "x": [[1., 2.], [30, 4.]],
        "y": [1., 2.],
    }


def test_column_collator_colmajor():
    rows = {"x": [1., 2.], "y": [1., 2.]}
    collator = ColumnCollator()
    batch = collator(rows)
    assert batch == rows


def test_column_collator_custom():
    collator = ColumnCollator(
        collators={
            "a": sum,
        }
    )
    batch = collator(
        [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
        ]
    )
    assert batch["a"] == 3
    assert batch["b"] == ["x", "y"]
