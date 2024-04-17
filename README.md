# UDL Mini-Project
## Candidate Number: 1068811

---

## Comparing Generative Models for Generative Replay in Bayesian Continual Learning

The experiments for this project were run on a 24GB NVIDIA GeForce RTX 3090.

The Variational Continual Learning implementation found in this repository is based on the following publicly available repositories:
- https://github.com/nvcuong/variational-continual-learning
- https://github.com/NixGD/variational-continual-learning

The pretrained diffusion models can be found at:
- MNIST: https://huggingface.co/gnokit/unet-mnist-32
- CIFAR10: https://huggingface.co/google/ddpm-cifar10-32

The DCGAN implementation is based on the following implementation:
- https://github.com/pytorch/examples/tree/main/dcgan

The CIFAR10 classifier can be found at:
- https://github.com/aaron-xichen/pytorch-playground

---

![Perm MNIST](/results/perm_mnist/plot.png "Perm MNIST")

![Single-head Split MNIST](/results/split_mnist/plot.png "Single-head Split MNIST")

![Single-head CIFAR10](/results/cifar10/plot.png "Single-head CIFAR10")
