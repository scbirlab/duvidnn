def test_move_to_device():
    import torch

    from duvidnn.utils.device import move_to_device

    batch = {
        "x": torch.ones(2, 3),
        "nested": {
            "y": torch.zeros(2, 1),
        },
        "other": [
            torch.ones(1),
            "unchanged",
        ],
    }

    observed = move_to_device(
        batch,
        "cpu",
    )

    assert observed["x"].device.type == "cpu"
    assert observed["nested"]["y"].device.type == "cpu"
    assert observed["other"][0].device.type == "cpu"
    assert observed["other"][1] == "unchanged"