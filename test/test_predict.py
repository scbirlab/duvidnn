def test_box_predict_preserves_training_pipeline():
    import torch
    from torch import nn

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap

    pipeline = DataPipeline({
        "x": ("x_raw", "identity"),
    })

    box = Box(
        model=nn.Linear(2, 1),
        pipeline=pipeline,
        input_map=ColumnMap(
            inputs={"input": "x"},
        ),
    )
    training = {
        "x_raw": [
            [1., 2.],
            [3., 4.],
        ],
    }
    box.prepare(training)

    training_data = box.pipeline.data_out

    predictions = box.predict({
        "x_raw": [
            [5., 6.],
            [7., 8.],
            [9., 10.],
        ],
    })

    assert box.pipeline.data_out is training_data
    assert "prediction" in predictions.column_names
    assert len(predictions) == 3
