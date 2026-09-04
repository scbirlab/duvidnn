from duvidnn.model.cnn import CNN2D

def test_cnn2d():

    model = CNN2D(
        input_channels=3,
        channels=[8, 16],
        hidden_dims=32,
        output_dim=2,
    )

    observed = model(
        torch.randn(
            4,
            3,
            32,
            32,
        )
    )

    assert observed.shape == (4, 2)
