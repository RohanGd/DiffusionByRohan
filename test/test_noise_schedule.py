
def test_noise_schedule():
    import torch
    from model import UniformNoiseSchedule

    noise_scheduler = UniformNoiseSchedule(T=10)

    t = torch.arange(2).reshape(2, 1)

    assert noise_scheduler(t).shape == t.shape

    assert noise_scheduler.bar(t).shape == t.shape
