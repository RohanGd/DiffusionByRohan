
To run tests run `pytest`; for debugger mode run `pytest --pdb` press c to continue from breakpoints.


### Training

1. Sample a clean image $x_0 \sim q_0(x_0)$ and a noise $\epsilon \sim \mathcal{N}(0,1)$ and a time step $ t \sim \mathcal{U}(1,T)$.

   Then create the noised image: 
   $$
      x_t = \sqrt{\bar{\alpha _t}} x_0 + \sqrt{1-\bar{\alpha _t}} \epsilon
   $$

2. Have the neural network return us $\epsilon _\theta$ based on $x_t$ and $t$ as inputs; where $\epsilon _\theta$ is the approximation of the true noise $\epsilon$ that was added.

3. Compute Loss $\mathcal{L} = ||\epsilon _\theta(x_t, t) - \epsilon||^2$ and backpropogate through $\epsilon _\theta(x_t, t)$ optimizing $\theta$.


### Inference

1. Sample noise $x_t \sim \mathcal{N}(0,1)$.
2. Perform Iterative update from $x_t$ to $x_{t-1}$ using:
$$
   x_{t-1} = \frac{1}{\sqrt{\alpha_t}} . \{x_t - \frac{1 -\alpha_t}{\sqrt{1-\bar{\alpha_t}}} \epsilon_\theta(x_t, t) \} + \sigma_t.z
$$






## Observations on backbone:
1. Simple CNN
2. Using Sigmoid vs tanh
3. Adding residuals
4. Batch size
5. Kernel size
6. Unet vs CNN