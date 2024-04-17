"""
Permuted MNIST experiments.
"""

import os
import argparse
from tqdm import tqdm

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision.datasets import MNIST

from diffusers import UNet2DModel, DDIMScheduler

from models import Vanilla_NN, MFVI_NN, FC_Generator, FC_Discriminator
from coreset import RandomCoreset, KCenterCoreset
from pipeline_ddim import DDIMPipeline
from train_unet import train_unet

import warnings
warnings.filterwarnings("ignore")

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def perm_mnist_dataset(n_tasks):
    random_perms = [torch.randperm(784) for _ in range(n_tasks)]
    mnist_transforms = [transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)[random_perms[i]]),
        ]
    ) for i in range(n_tasks)]

    train_datasets = ConcatDataset([MNIST(root="~/.torch/data/mnist", train=True, download=True, transform=t) for t in mnist_transforms])
    task_size = len(train_datasets) // n_tasks
    train_task_ids = torch.cat([torch.full((task_size,), id) for id in range(n_tasks)])

    test_datasets = ConcatDataset([MNIST(root="~/.torch/data/mnist", train=False, download=True, transform=t) for t in mnist_transforms])
    task_size = len(test_datasets) // n_tasks
    test_task_ids = torch.cat([torch.full((task_size,), id) for id in range(n_tasks)])

    return train_datasets, train_task_ids, test_datasets, test_task_ids

def run_vcl_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, coreset, device):
    print("Running VCL experiment")
    
    # Instantiate model for posterior initialisation
    vanilla_model = Vanilla_NN(
        input_size=784,
        hidden_size=opt.hidden_size,
        output_size=opt.n_classes,
        n_hidden_layers=opt.n_hidden_layers,
    ).to(device)

    # Instantiate optimizer and loss function
    optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=opt.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_task_indices = torch.where(train_task_ids == 0)[0]
    task_data = torch.utils.data.Subset(train_datasets, train_task_indices)
    train_task_loader = DataLoader(task_data, batch_size=opt.batch_size)

    for _ in tqdm(range(opt.epochs), desc="Posterior Init | Epochs"):
        for x, y in train_task_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = vanilla_model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
    mf_weights = vanilla_model.get_weights()

    mfvi_model = MFVI_NN(
        input_size=784,
        hidden_size=opt.hidden_size,
        output_size=opt.n_classes,
        n_hidden_layers=opt.n_hidden_layers,
        n_heads=1,
        prev_weights=mf_weights
    ).to(device)

    mean_accuracies = []
    for task_id in range(opt.n_tasks):

        # Instantiate a separate optimizer for each task
        optimizer = torch.optim.Adam(mfvi_model.parameters(), lr=opt.lr)

        # Instantiate a separate dataloader for each task
        train_task_indices = torch.where(train_task_ids == task_id)[0]
        task_data = torch.utils.data.Subset(train_datasets, train_task_indices)
        if coreset is not None:
            non_coreset_train_data = coreset.select(task_data, task_id)
            train_task_loader = DataLoader(non_coreset_train_data, batch_size=opt.batch_size, shuffle=True)
        else:
            train_task_loader = DataLoader(task_data, batch_size=opt.batch_size, shuffle=True)

        for _ in tqdm(range(opt.epochs), desc=f"Task {task_id} | Epochs"):

            # Training loop on non-coreset data to optimise posterior
            mfvi_model.train()
            for x, y in train_task_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = mfvi_model.vcl_loss(x, y, head=0, task_size=len(task_data), n_train_samples=opt.train_number_samples)
                loss.backward()
                optimizer.step()

        # Reset model for new task
        mfvi_model.reset_for_new_task(head=0)

        if opt.train_full_coreset and coreset is not None:
            mfvi_model = coreset.coreset_train(
                mfvi_model, optimizer, opt.lr, list(range(task_id + 1)), opt.epochs, device, y_transform=None, multiheaded=False, batch_size=opt.batch_size
            )

        # Test model
        task_accuracies = []
        total_accuracy = 0
        total_test_samples = 0
        for test_task_id in range(task_id + 1):

            if not opt.train_full_coreset and coreset is not None:
                mfvi_model = coreset.coreset_train(
                    mfvi_model, optimizer, opt.lr, test_task_id, opt.epochs, device, y_transform=None, multiheaded=False, batch_size=opt.batch_size
                )

            test_task_indices = torch.where(test_task_ids == test_task_id)[0]
            task_data = Subset(test_datasets, test_task_indices)
            x = torch.stack([x for x, _ in task_data]) # [bsz, 784]
            y = torch.stack([torch.tensor(y) for _, y in task_data]) # [bsz]
            x, y = x.to(device), y.to(device)

            outputs = []
            mfvi_model.eval()
            with torch.no_grad():
                for _ in range(opt.test_number_samples):
                    out = mfvi_model(x, 0) # [bsz, n_classes]
                    out = F.softmax(out, dim=1)
                    outputs.append(out)
            outputs = torch.stack(outputs).mean(dim=0)
            y_pred = torch.argmax(outputs, dim=1)
            acc = (y_pred == y).sum().item() / y.size(0)
            print("After task {} perfomance on task {} is {}".format(task_id, test_task_id, acc))
            task_accuracies.append(acc)
            total_accuracy += acc * len(task_data)
            total_test_samples += len(task_data)
        
        mean_accuracy = total_accuracy / total_test_samples
        mean_accuracies.append(mean_accuracy)
        print(f"{task_id + 1} Tasks - Mean accuracy: {mean_accuracy}")

        # Save model
        os.makedirs("mfvi-models/perm_mnist", exist_ok=True)
        if coreset is None:
            torch.save(mfvi_model.state_dict(), f"mfvi-models/perm_mnist/vcl_task_{task_id}.pt")
        else:
            torch.save(mfvi_model.state_dict(), f"mfvi-models/perm_mnist/vcl_coreset_{opt.coreset_alg}_task_{task_id}.pt")

    # Reset pytorch
    torch.cuda.empty_cache()
    del vanilla_model, mfvi_model, optimizer, loss_fn

    # Save results
    os.makedirs("results/perm_mnist", exist_ok=True)
    if coreset is None:
        np.save(f"results/perm_mnist/vcl.npy", mean_accuracies)
    else:
        np.save(f"results/perm_mnist/vcl_coreset_{opt.coreset_alg}.npy", mean_accuracies)

def train_gan(train_dataloader, epochs, device):
    generator = FC_Generator(784, 784)
    generator = generator.to(device)
    discriminator = FC_Discriminator(784, 1)
    discriminator = discriminator.to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    loss_fn = nn.BCELoss()

    for _ in tqdm(range(epochs), desc="GAN | Epochs"):
        for x, _ in train_dataloader:
            x = x.to(device)
            batch_size = x.size(0)
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # Train discriminator
            discriminator.zero_grad()
            real_output = discriminator(x)
            real_loss = loss_fn(real_output, real_label)
            real_loss.backward()

            z = torch.randn(batch_size, 784).to(device)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            fake_loss = loss_fn(fake_output, fake_label)
            fake_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            generator.zero_grad()
            fake_output = discriminator(fake_data)
            generator_loss = loss_fn(fake_output, real_label)
            generator_loss.backward()
            generator_optimizer.step()
    
    return generator

def run_vgr_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, device):
    print("Running VGR (GAN) experiment")
    
    # Create directory to save GANs
    os.makedirs("vgr-gan/perm_mnist", exist_ok=True)

    # Instantiate model for posterior initialisation
    vanilla_model = Vanilla_NN(
        input_size=784,
        hidden_size=opt.hidden_size,
        output_size=opt.n_classes,
        n_hidden_layers=opt.n_hidden_layers,
    ).to(device)

    # Instantiate optimizer and loss function
    optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=opt.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_task_indices = torch.where(train_task_ids == 0)[0]
    task_data = torch.utils.data.Subset(train_datasets, train_task_indices)
    train_task_loader = DataLoader(task_data, batch_size=opt.batch_size)

    for _ in tqdm(range(opt.epochs), desc="Posterior Init | Epochs"):
        for x, y in train_task_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = vanilla_model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
    mf_weights = vanilla_model.get_weights()

    mfvi_model = MFVI_NN(
        input_size=784,
        hidden_size=opt.hidden_size,
        output_size=opt.n_classes,
        n_hidden_layers=opt.n_hidden_layers,
        n_heads=1,
        prev_weights=mf_weights
    ).to(device)

    mean_accuracies = []
    for task_id in range(opt.n_tasks):

        if task_id != 0:

            # Instantiate a separate dataloader for each task
            train_task_indices = torch.where(train_task_ids == task_id)[0]
            real_task_data = Subset(train_datasets, train_task_indices)
            x_real = torch.stack([x for x, _ in real_task_data]) # [bsz, 784]
            y_real = torch.stack([torch.tensor(y) for _, y in real_task_data]) # [bsz]
            real_task_data = TensorDataset(x_real, y_real)

            # Sample from previous GAN
            generated_dataset = []
            for _task in range(task_id):
                for cls in range(opt.n_classes):
                    generator = FC_Generator(784, 784)
                    generator.load_state_dict(torch.load(f"vgr-gan/perm_mnist/generator_task_{_task}_class_{cls}.pt"))
                    generator = generator.to(device)
                    generator.eval()
                    z = torch.randn((6000, 784)).to(device)
                    generated_data = generator(z).detach().cpu()
                    generated_data_labels = torch.full((6000,), cls)
                    generated_dataset.append(TensorDataset(generated_data, generated_data_labels))

            generated_dataset.append(real_task_data)
            generated_dataset = ConcatDataset(generated_dataset)
            train_task_loader = DataLoader(generated_dataset, batch_size=opt.batch_size * (task_id + 1), shuffle=True)

            # Instantiate a separate optimizer for each task
            optimizer = torch.optim.Adam(mfvi_model.parameters(), lr=opt.lr)

            for _ in tqdm(range(opt.epochs), desc=f"Task {task_id} | Epochs"):
                mfvi_model.train()
                for x, y in train_task_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = mfvi_model.vcl_loss(x, y, head=0, task_size=len(task_data), n_train_samples=opt.train_number_samples)
                    loss.backward()
                    optimizer.step()

            # Reset model for new task
            mfvi_model.reset_for_new_task(head=0)

            # Test model
            task_accuracies = []
            total_accuracy = 0
            total_test_samples = 0
            for test_task_id in range(task_id + 1):

                test_task_indices = torch.where(test_task_ids == test_task_id)[0]
                task_data = Subset(test_datasets, test_task_indices)
                x = torch.stack([x for x, _ in task_data]) # [bsz, 784]
                y = torch.stack([torch.tensor(y) for _, y in task_data]) # [bsz]
                x, y = x.to(device), y.to(device)

                outputs = []
                mfvi_model.eval()
                with torch.no_grad():
                    for _ in range(opt.test_number_samples):
                        out = mfvi_model(x, 0) # [bsz, n_classes]
                        out = F.softmax(out, dim=1)
                        outputs.append(out)
                outputs = torch.stack(outputs).mean(dim=0)
                y_pred = torch.argmax(outputs, dim=1)
                acc = (y_pred == y).sum().item() / y.size(0)
                print("After task {} perfomance on task {} is {}".format(task_id, test_task_id, acc))
                task_accuracies.append(acc)
                total_accuracy += acc * len(task_data)
                total_test_samples += len(task_data)
        
            mean_accuracy = total_accuracy / total_test_samples
            mean_accuracies.append(mean_accuracy)
            print(f"{task_id + 1} Tasks - Mean accuracy: {mean_accuracy}")

            # Train GAN
            for cls in range(opt.n_classes):
                y = torch.tensor([y for _, y in real_task_data])
                class_indices = torch.where(y == cls)[0]
                class_data = Subset(real_task_data, class_indices)
                class_loader = DataLoader(class_data, batch_size=opt.batch_size, shuffle=True)
                generator = train_gan(class_loader, opt.gan_epochs, device)
                torch.save(generator.state_dict(), f"vgr-gan/perm_mnist/generator_task_{task_id}_class_{cls}.pt")
            
            # Save model
            torch.save(mfvi_model.state_dict(), f"mfvi-models/perm_mnist/vgr_gan_task_{task_id}.pt")

        else:

            # Instantiate a separate dataloader for each task
            train_task_indices = torch.where(train_task_ids == task_id)[0]
            task_data = Subset(train_datasets, train_task_indices)
            train_task_loader = DataLoader(task_data, batch_size=opt.batch_size, shuffle=True)

            # Instantiate a separate optimizer for each task
            optimizer = torch.optim.Adam(mfvi_model.parameters(), lr=opt.lr)

            for _ in tqdm(range(opt.epochs), desc=f"Task {task_id} | Epochs"):
                mfvi_model.train()
                for x, y in train_task_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = mfvi_model.vcl_loss(x, y, head=0, task_size=len(task_data), n_train_samples=opt.train_number_samples)
                    loss.backward()
                    optimizer.step()
                
            # Reset model for new task
            mfvi_model.reset_for_new_task(head=0)

            # Test model
            task_accuracies = []
            total_accuracy = 0
            total_test_samples = 0
            for test_task_id in range(task_id + 1):

                test_task_indices = torch.where(test_task_ids == test_task_id)[0]
                task_data = Subset(test_datasets, test_task_indices)
                x = torch.stack([x for x, _ in task_data]) # [bsz, 784]
                y = torch.stack([torch.tensor(y) for _, y in task_data]) # [bsz]
                x, y = x.to(device), y.to(device)

                outputs = []
                mfvi_model.eval()
                with torch.no_grad():
                    for _ in range(opt.test_number_samples):
                        out = mfvi_model(x, 0) # [bsz, n_classes]
                        out = F.softmax(out, dim=1)
                        outputs.append(out)
                outputs = torch.stack(outputs).mean(dim=0)
                y_pred = torch.argmax(outputs, dim=1)
                acc = (y_pred == y).sum().item() / y.size(0)
                print("After task {} perfomance on task {} is {}".format(task_id, test_task_id, acc))
                task_accuracies.append(acc)
                total_accuracy += acc * len(task_data)
                total_test_samples += len(task_data)
        
            mean_accuracy = total_accuracy / total_test_samples
            mean_accuracies.append(mean_accuracy)
            print(f"{task_id + 1} Tasks - Mean accuracy: {mean_accuracy}")

            # Train GAN
            for cls in range(opt.n_classes):
                y = torch.tensor([y for _, y in task_data])
                class_indices = torch.where(y == cls)[0]
                class_data = Subset(task_data, class_indices)
                class_loader = DataLoader(class_data, batch_size=opt.batch_size, shuffle=True)
                generator = train_gan(class_loader, opt.gan_epochs, device)
                torch.save(generator.state_dict(), f"vgr-gan/perm_mnist/generator_task_{task_id}_class_{cls}.pt")

            # Save model
            os.makedirs("mfvi-models/perm_mnist", exist_ok=True)
            torch.save(mfvi_model.state_dict(), f"mfvi-models/perm_mnist/vgr_gan_task_{task_id}.pt")

    # Reset pytorch
    del vanilla_model, mfvi_model, optimizer, loss_fn
    torch.cuda.empty_cache()

    # Save results
    os.makedirs("results/perm_mnist", exist_ok=True)
    np.save(f"results/perm_mnist/vgr_gan.npy", mean_accuracies)


def run_vgr_dm_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, device):
    print("Running VGR (DM) experiment")
    
    # Create directory to save GANs
    os.makedirs("vgr-unet/perm_mnist", exist_ok=True)

    # Instantiate model for posterior initialisation
    vanilla_model = Vanilla_NN(
        input_size=784,
        hidden_size=opt.hidden_size,
        output_size=opt.n_classes,
        n_hidden_layers=opt.n_hidden_layers,
    ).to(device)

    # Instantiate optimizer and loss function
    optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=opt.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_task_indices = torch.where(train_task_ids == 0)[0]
    task_data = torch.utils.data.Subset(train_datasets, train_task_indices)
    train_task_loader = DataLoader(task_data, batch_size=opt.batch_size)

    for _ in tqdm(range(opt.epochs), desc="Posterior Init | Epochs"):
        for x, y in train_task_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = vanilla_model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
    mf_weights = vanilla_model.get_weights()

    mfvi_model = MFVI_NN(
        input_size=784,
        hidden_size=opt.hidden_size,
        output_size=opt.n_classes,
        n_hidden_layers=opt.n_hidden_layers,
        n_heads=1,
        prev_weights=mf_weights
    ).to(device)

    mean_accuracies = []
    previous_tasks_sampled_dataset = []
    for task_id in range(opt.n_tasks):

        if task_id != 0:

            # Instantiate a separate dataloader for each task
            train_task_indices = torch.where(train_task_ids == task_id)[0]
            real_task_data = Subset(train_datasets, train_task_indices)
            x_real = torch.stack([x for x, _ in real_task_data]) # [bsz, 784]
            y_real = torch.stack([torch.tensor(y) for _, y in real_task_data]) # [bsz]
            real_task_data = TensorDataset(x_real, y_real)

            # Sample from DM for task: task_id - 1
            for cls in range(opt.n_classes):
                print(f"Sampling - Task {task_id - 1} | Class {cls}")
                unet = UNet2DModel(sample_size=32, in_channels=1, out_channels=1, block_out_channels=(32,64,128,256), norm_num_groups=8)
                unet.load_state_dict(torch.load(f"vgr-unet/perm_mnist/unet_task_{task_id - 1}_class_{cls}.pt"))
                scheduler = DDIMScheduler(num_train_timesteps=500)
                ddim_pipeline = DDIMPipeline(unet, scheduler)
                ddim_pipeline.enable_xformers_memory_efficient_attention()
                ddim_pipeline.to(device)
                ddim_pipeline.set_progress_bar_config(disable=True)
                generated_data = []
                for _ in range(6):
                    generated_data.append(ddim_pipeline(batch_size=1000, num_inference_steps=opt.num_inference_steps))
                generated_data = torch.cat(generated_data, dim=0)
                generated_data_labels = torch.full((6000,), cls)
                previous_tasks_sampled_dataset.append(TensorDataset(generated_data, generated_data_labels))
                del unet, ddim_pipeline
                torch.cuda.empty_cache()

            generated_dataset = []
            generated_dataset.append(real_task_data)
            generated_dataset.extend(previous_tasks_sampled_dataset)
            generated_dataset = ConcatDataset(generated_dataset)
            train_task_loader = DataLoader(generated_dataset, batch_size=opt.batch_size * (task_id + 1), shuffle=True)

            # Instantiate a separate optimizer for each task
            optimizer = torch.optim.Adam(mfvi_model.parameters(), lr=opt.lr)

            for _ in tqdm(range(opt.epochs), desc=f"Task {task_id} | Epochs"):
                mfvi_model.train()
                for x, y in train_task_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = mfvi_model.vcl_loss(x, y, head=0, task_size=len(task_data), n_train_samples=opt.train_number_samples)
                    loss.backward()
                    optimizer.step()

            # Reset model for new task
            mfvi_model.reset_for_new_task(head=0)

            # Test model
            task_accuracies = []
            total_accuracy = 0
            total_test_samples = 0
            for test_task_id in range(task_id + 1):

                test_task_indices = torch.where(test_task_ids == test_task_id)[0]
                task_data = Subset(test_datasets, test_task_indices)
                x = torch.stack([x for x, _ in task_data]) # [bsz, 784]
                y = torch.stack([torch.tensor(y) for _, y in task_data]) # [bsz]
                x, y = x.to(device), y.to(device)

                outputs = []
                mfvi_model.eval()
                with torch.no_grad():
                    for _ in range(opt.test_number_samples):
                        out = mfvi_model(x, 0) # [bsz, n_classes]
                        out = F.softmax(out, dim=1)
                        outputs.append(out)
                outputs = torch.stack(outputs).mean(dim=0)
                y_pred = torch.argmax(outputs, dim=1)
                acc = (y_pred == y).sum().item() / y.size(0)
                print("After task {} perfomance on task {} is {}".format(task_id, test_task_id, acc))
                task_accuracies.append(acc)
                total_accuracy += acc * len(task_data)
                total_test_samples += len(task_data)
        
            mean_accuracy = total_accuracy / total_test_samples
            mean_accuracies.append(mean_accuracy)
            print(f"{task_id + 1} Tasks - Mean accuracy: {mean_accuracy}")

            # Train UNet
            for cls in range(opt.n_classes):
                y = torch.tensor([y for _, y in real_task_data])
                class_indices = torch.where(y == cls)[0]
                class_data = Subset(real_task_data, class_indices)
                train_unet(opt, class_data, task_id=task_id, cls=cls)
                torch.cuda.empty_cache()

            # Save model
            torch.save(mfvi_model.state_dict(), f"mfvi-models/perm_mnist/vgr_ddpm_task_{task_id}.pt")

        else:

            # Instantiate a separate dataloader for each task
            train_task_indices = torch.where(train_task_ids == task_id)[0]
            task_data = Subset(train_datasets, train_task_indices)
            train_task_loader = DataLoader(task_data, batch_size=opt.batch_size, shuffle=True)

            # Instantiate a separate optimizer for each task
            optimizer = torch.optim.Adam(mfvi_model.parameters(), lr=opt.lr)

            for _ in tqdm(range(opt.epochs), desc=f"Task {task_id} | Epochs"):
                mfvi_model.train()
                for x, y in train_task_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = mfvi_model.vcl_loss(x, y, head=0, task_size=len(task_data), n_train_samples=opt.train_number_samples)
                    loss.backward()
                    optimizer.step()
                
            # Reset model for new task
            mfvi_model.reset_for_new_task(head=0)

            # Test model
            task_accuracies = []
            total_accuracy = 0
            total_test_samples = 0
            for test_task_id in range(task_id + 1):

                test_task_indices = torch.where(test_task_ids == test_task_id)[0]
                task_data = Subset(test_datasets, test_task_indices)
                x = torch.stack([x for x, _ in task_data]) # [bsz, 784]
                y = torch.stack([torch.tensor(y) for _, y in task_data]) # [bsz]
                x, y = x.to(device), y.to(device)

                outputs = []
                mfvi_model.eval()
                with torch.no_grad():
                    for _ in range(opt.test_number_samples):
                        out = mfvi_model(x, 0) # [bsz, n_classes]
                        out = F.softmax(out, dim=1)
                        outputs.append(out)
                outputs = torch.stack(outputs).mean(dim=0)
                y_pred = torch.argmax(outputs, dim=1)
                acc = (y_pred == y).sum().item() / y.size(0)
                print("After task {} perfomance on task {} is {}".format(task_id, test_task_id, acc))
                task_accuracies.append(acc)
                total_accuracy += acc * len(task_data)
                total_test_samples += len(task_data)
        
            mean_accuracy = total_accuracy / total_test_samples
            mean_accuracies.append(mean_accuracy)
            print(f"{task_id + 1} Tasks - Mean accuracy: {mean_accuracy}")

            # Train UNet
            for cls in range(opt.n_classes):
                y = torch.tensor([y for _, y in task_data])
                class_indices = torch.where(y == cls)[0]
                class_data = Subset(task_data, class_indices)
                train_unet(opt, class_data, task_id=task_id, cls=cls)
                torch.cuda.empty_cache()

            # Save model
            os.makedirs("mfvi-models/perm_mnist", exist_ok=True)
            torch.save(mfvi_model.state_dict(), f"mfvi-models/perm_mnist/vgr_ddpm_task_{task_id}.pt")

    # Reset pytorch
    torch.cuda.empty_cache()

    # Save results
    os.makedirs("results/perm_mnist", exist_ok=True)
    np.save(f"results/perm_mnist/vgr_ddpm.npy", mean_accuracies)


def main():
    parser = argparse.ArgumentParser(description='Permuted MNIST experiments.')
    parser.add_argument('experiment', type=str, choices=['all', 'vcl', 'vcl-coreset', 'vgr-gan', 'vgr-dm'], help='Experiment to run. One of "all", "vcl", "vcl-coreset", "vgr-gan", "vgr-dm".')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of MNIST classes. Determines model output size.')
    parser.add_argument('--hidden_size', type=int, default=100, help='Size of hidden layers.')
    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--n_tasks', type=int, default=10, help='Number of tasks.')
    parser.add_argument('--coreset_alg', type=str, default='kcenter', choices=['random', 'kcenter'], help='Algorithm to use for coreset selection.')
    parser.add_argument('--coreset_size', type=int, default=200, help='Size of the coreset.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--train_full_coreset', type=bool, default=True, help='Whether to train on the full coreset after each task.')
    parser.add_argument('--train_number_samples', type=int, default=10, help='Number of times to sample the weights and biases of the model during training.')
    parser.add_argument('--test_number_samples', type=int, default=50, help='Number of times to sample the weights and biases of the model during testing.')
    
    parser.add_argument('--gan_epochs', type=int, default=200, help='Number of epochs to train the GAN.')

    parser.add_argument('--unet_epochs', type=int, default=100, help='Number of epochs to train the UNet.')
    parser.add_argument('--unet_batch_size', type=int, default=256, help='Batch size for training the UNet.')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps for diffusion model.')

    opt = parser.parse_args()

    # Set seed for reproducible results
    configure_seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_datasets, train_task_ids, test_datasets, test_task_ids = perm_mnist_dataset(opt.n_tasks)

    if opt.experiment == "all":

        # VCL without coreset
        run_vcl_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, None, device)

        # VCL with coreset
        if opt.coreset_alg == 'random':
            coreset = RandomCoreset(opt.coreset_size)
        elif opt.coreset_alg == 'kcenter':
            coreset = KCenterCoreset(opt.coreset_size)
        else:
            raise ValueError(f"Unknown coreset algorithm: {opt.coreset_alg}")
        run_vcl_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, coreset, device)

        # VGR (GAN)
        run_vgr_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, device)

        # VGR (DDPM)
        run_vgr_dm_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, device)
    
    elif opt.experiment == "vcl":
            
        # VCL without coreset
        run_vcl_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, None, device)

    elif opt.experiment == "vcl-coreset":

        # VCL with coreset
        if opt.coreset_alg == 'random':
            coreset = RandomCoreset(opt.coreset_size)
        elif opt.coreset_alg == 'kcenter':
            coreset = KCenterCoreset(opt.coreset_size)
        else:
            raise ValueError(f"Unknown coreset algorithm: {opt.coreset_alg}")
        run_vcl_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, coreset, device)

    elif opt.experiment == "vgr-gan":

        # VGR (GAN)
        run_vgr_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, device)
    
    elif opt.experiment == "vgr-dm":
            
        # VGR (LDM)
        run_vgr_dm_experiment(opt, train_datasets, train_task_ids, test_datasets, test_task_ids, device)

    else:
        raise ValueError(f"Unknown experiment: {opt.experiment}")

if __name__ == '__main__':
    main()