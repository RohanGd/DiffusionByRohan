def test_forward_diffusion_process():
    import torch
    from torchvision.io import read_image
    import torch.nn.functional as F

    from model import UniformNoiseSchedule
    from dataload.forwardDiffusionProcess import forward_diffusion_process
    import matplotlib.pyplot as plt
    T = 10
    noise_scheduler = UniformNoiseSchedule(T)

    img_path = "data/smiley.jpg"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x0 = read_image(img_path)
    x0 = F.interpolate(x0.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
    x0 = x0.to(dtype=torch.float32, device=device)

    noise_scheduler = noise_scheduler.to(device=device)
    t = torch.tensor([1], device =device, dtype=torch.long)
    
    x1, noise_1 = forward_diffusion_process(
        noise_scheduler=noise_scheduler, x0=x0, t=t
    )

    assert x1.shape == noise_1.shape and x1.shape == x0.shape
    alpha_bar_t = noise_scheduler.bar(t)
    alpha_t = noise_scheduler(t)


    x0_rev_from_x1 = (1/torch.sqrt(alpha_t)) * (x1 - ((1-alpha_t) / torch.sqrt(1-alpha_bar_t)) * noise_1)
    assert x0_rev_from_x1.shape == x1.shape
    assert torch.isclose(x0, x0_rev_from_x1, rtol=1e-4, atol=1e-4).all().item() == True


    x2, noise_2 = forward_diffusion_process(
        noise_scheduler=noise_scheduler, x0=x1, t=t
    )
    
    alpha_bar_t = noise_scheduler.bar(t)
    alpha_t = noise_scheduler(t)
    x1_rev_from_x2 = (1/torch.sqrt(alpha_t)) * (x2 - ((1-alpha_t) / torch.sqrt(1-alpha_bar_t)) * noise_2)
    assert torch.isclose(x1, x1_rev_from_x2).all().item() == True


    x0_rev_from_x1_rev = (1/torch.sqrt(alpha_t)) * (x1_rev_from_x2 - ((1-alpha_t) / torch.sqrt(1-alpha_bar_t)) * noise_1)
    assert torch.isclose(x0, x0_rev_from_x1_rev, rtol=1e-4, atol=1e-4).all().item() == True
