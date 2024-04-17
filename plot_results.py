import os
import numpy as np
import matplotlib.pyplot as plt

# Permuted MNIST

plt.rcParams.update({'font.size': 16})

vcl = np.load("/home/miguel/workspace/udl/results/perm_mnist/vcl.npy")
coreset = np.load("/home/miguel/workspace/udl/results/perm_mnist/vcl_coreset_kcenter.npy")
gan = np.load("/home/miguel/workspace/udl/results/perm_mnist/vgr_gan.npy")
dm = np.load("/home/miguel/workspace/udl/results/perm_mnist/vgr_ddpm.npy")
num_tasks = np.arange(1, 11)

plt.figure(figsize=(8, 6))
plt.plot(num_tasks, vcl, label='VCL', marker='o')
plt.plot(num_tasks, coreset, label='VCL+core', marker='o')
plt.plot(num_tasks, gan, label='VGR-GAN', marker='o')
plt.plot(num_tasks, dm, label='VGR-DM', marker='o')

# Add labels and legend
plt.xlabel('Model after $n$ tasks')
plt.ylabel('Accuracy on seen tasks')
plt.yticks(np.arange(0.8, 1.01, 0.05))
plt.xticks(np.arange(1, 11, 1))
plt.legend()
plt.gca().spines['top'].set_visible(False)    # Show top spine
plt.gca().spines['right'].set_visible(False) # Hide right spine
plt.gca().spines['bottom'].set_visible(True) # Show bottom spine
plt.gca().spines['left'].set_visible(True)   # Show left spine

plt.tight_layout()

# Save the plot
plt.savefig("/home/miguel/workspace/udl/results/perm_mnist/plot.png")

# Split MNIST

plt.rcParams.update({'font.size': 16})

vcl = np.load("/home/miguel/workspace/udl/results/split_mnist/vcl.npy")
coreset = np.load("/home/miguel/workspace/udl/results/split_mnist/vcl_coreset_kcenter.npy")
gan = np.load("/home/miguel/workspace/udl/results/split_mnist/vgr_gan.npy")
dm = np.load("/home/miguel/workspace/udl/results/split_mnist/vgr_ddpm.npy")
num_tasks = np.arange(1, 6)

plt.figure(figsize=(8, 6))
plt.plot(num_tasks, vcl, label='VCL', marker='o')
plt.plot(num_tasks, coreset, label='VCL+core', marker='o')
plt.plot(num_tasks, gan, label='VGR-GAN', marker='o')
plt.plot(num_tasks, dm, label='VGR-DM', marker='o')

# Add labels and legend
plt.xlabel('Model after $n$ tasks')
plt.ylabel('Accuracy on seen tasks')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(1, 5.5, 1))
plt.legend()
plt.gca().spines['top'].set_visible(False)    # Show top spine
plt.gca().spines['right'].set_visible(False) # Hide right spine
plt.gca().spines['bottom'].set_visible(True) # Show bottom spine
plt.gca().spines['left'].set_visible(True)   # Show left spine

plt.tight_layout()

# Save the plot
plt.savefig("/home/miguel/workspace/udl/results/split_mnist/plot.png")

# CIFAR10

plt.rcParams.update({'font.size': 16})

vcl = np.load("/home/miguel/workspace/udl/results/cifar10/vcl.npy")
coreset = np.load("/home/miguel/workspace/udl/results/cifar10/vcl_coreset_kcenter.npy")
gan = np.load("/home/miguel/workspace/udl/results/cifar10/vgr_gan.npy")
dm = np.load("/home/miguel/workspace/udl/results/cifar10/vgr_ddpm.npy")
num_tasks = np.arange(1, 6)

plt.figure(figsize=(8, 6))
plt.plot(num_tasks, vcl, label='VCL', marker='o')
plt.plot(num_tasks, coreset, label='VCL+core', marker='o')
plt.plot(num_tasks, gan, label='VGR-GAN', marker='o')
plt.plot(num_tasks, dm, label='VGR-DM', marker='o')

# Add labels and legend
plt.xlabel('Model after $n$ tasks')
plt.ylabel('Accuracy on seen tasks')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(1, 5.5, 1))
plt.legend()
plt.gca().spines['top'].set_visible(False)    # Show top spine
plt.gca().spines['right'].set_visible(False) # Hide right spine
plt.gca().spines['bottom'].set_visible(True) # Show bottom spine
plt.gca().spines['left'].set_visible(True)   # Show left spine

plt.tight_layout()

# Save the plot
plt.savefig("/home/miguel/workspace/udl/results/cifar10/plot.png")
