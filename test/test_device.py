
import torch

from duvidnn.utils.device import move_to_device


def test_move_to_device():
    batch = {
        "x": torch.ones(2, 3),
        "nested": {"y": torch.zeros(2, 1)},
        "other": [
            torch.ones(1),
            "unchanged",
        ],
    }

    observed = move_to_device(batch, "cpu")

    assert observed["x"].device.type == "cpu"
    assert observed["nested"]["y"].device.type == "cpu"
    assert observed["other"][0].device.type == "cpu"
    assert observed["other"][1] == "unchanged"


def test_move_to_device_supports_inplace_to():

    class InPlace:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device

    value = InPlace()
    observed = move_to_device(value, torch.device("cpu"))

    assert observed is value
    assert observed.device == torch.device("cpu")


def test_move_to_device_uses_returned_value():

    class Returning:
        def to(self, device):
            return "moved"

    assert move_to_device(Returning(), torch.device("cpu")) == "moved"
