
def test_ddpm_forward_pass():
    import torch
    from model import DDPM, CNNNoiseEstimator, SinusoidalTimeEmbeddings, UniformNoiseSchedule

    B, C, H, W = 2, 3, 10, 10
    T = 5
    t = torch.arange(B).reshape((B,1))
    x_t = torch.randn((B, C, H, W))

    noise_schedule = UniformNoiseSchedule(T)
    time_embedder = SinusoidalTimeEmbeddings(C=64)
    noise_estimator = CNNNoiseEstimator(time_embedder=time_embedder, dim=(C, H, W))

    diffusion_model = DDPM(T=T, dim=(C, H, W), noise_schedule=noise_schedule, time_embedder=time_embedder, noise_estimator=noise_estimator)

    assert x_t.shape == diffusion_model(x_t, t).shape

    sample = diffusion_model.sample((B, C, H, W))
    assert sample.shape == x_t.shape
    assert torch.isfinite(sample).all()
