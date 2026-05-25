## Why is U-Net so powerful for diffusion compared to a simple CNN?

In this project, I implemented two different architectures to estimate the noise added to an image: `CNNNoiseEstimator` and `UNetNoiseEstimator`. 

This follows the standard training pipeline for diffusion models. The network learns to estimate the noise added over $t$ time steps. Once trained, we use this predicted noise to reverse the process, stepping backward from a noisy image to a clean one ($x_t \rightarrow x_{t-1}$).

### Model Scale Comparison
Despite the vast difference in output quality, the U-Net architecture actually uses fewer parameters than the baseline CNN models:

* **CNNNoiseEstimator:** 4,878,083 parameters
* **CNNNoiseEstimator_v2:** 4,882,691 parameters
* **UNetNoiseEstimator:** 4,514,563 parameters

### Architecture Setups
* **CNN Baseline:** A `nn.ModuleList` containing 8 stacked convolutional layers. During the forward pass, a time embedding is added before each convolution, followed by a residual connection.
* **CNN v2:** An upgraded baseline featuring `GroupNorm` layers and `SiLU` activations instead of `Sigmoid`/`Tanh`.
* **U-Net:** A standard encoder-decoder structure with skip connections. (This was AI Generated as I didnt care about learning to build that from bottom up)

---

### Qualitative Results Matrix


| Architecture | 1 Epoch | 5 Epochs | 10 Epochs |
| :--- | :---: | :---: | :---: |
| **CNN (v1)** | — | — | ![CNN 10 epochs](samples_cnn/epoch_0014.png) |
| **CNN (v2 with GroupNorm)** | — | — | ![CNN_v2 10 epochs](samples_cnn_group_norm/epoch_0014.png) |
| **U-Net** | ![UNet 1 epoch](samples_unet/epoch_0000.png) | ![UNet 5 epochs](samples_unet/epoch_0004.png) | — |

While the first CNN managed to produce a faint smiley face, the v2 variant unexpectedly failed to generate a clear structure. Meanwhile, **the U-Net architecture shows clear superiority, delivering higher output quality in just 1 to 5 epochs than the CNNs achieved in 10.**

---

## Now the question is why is the UNet so superior?

A plain CNN struggles because it processes everything at one fixed resolution. U-Net is superior because you  can see features at multiple scales and skip connection

A plain CNN denoises mostly through repeated local filtering, while a U-Net denoises by combining global context with precise spatial detail. That is why it converges faster and produces much better samples in your experiments.

Generally, U-Nets are the default choice when dealing with functions where inputs and outputs have the same size/shape (such as in image segmentation) due to the lack of a real information bottleneck. Thus, those are a natural good choice when dealing with diffusion models, where the network must predict the residual noise. Now they use DiT. For example, Stable Diffusion 3.5.