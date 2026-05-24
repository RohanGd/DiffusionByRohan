
def test_cnn_noise_estimator():
    import torch
    from model import CNNNoiseEstimator, SinusoidalTimeEmbeddings

    B = 2
    C = 3
    H = 10
    W = 10
    xt = torch.randn((B, C, H, W))
    t = torch.arange(B).reshape((B, 1))
    time_embedder = SinusoidalTimeEmbeddings(C = 64)

    noise_estimator = CNNNoiseEstimator(time_embedder=time_embedder, dim = (C, H, W))

    assert xt.shape == noise_estimator(xt, t).shape

    pass

