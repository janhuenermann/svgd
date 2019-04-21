## Implementation of Stein Variational Gradient Descent

The paper titled _Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm_ ([link](https://arxiv.org/pdf/1608.04471)) describes how one can approximate a given distribution by its log probability using samples.

In this implementation I created a _neural sampler_ that learns to sample from a 
given (unnormalized) log probability function, that could also be an energy function.

<figure>
<img src="anim.gif" style="width: 100%; max-width=480px">
<figcaption>Figure 1: Convergence of a neural network learning to sample from a gaussian mixture model.</figcaption>
</figure>
