from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler

def train_unet(opt, dataset, task_id, cls):

    unet_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.view(1, 28, 28)),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )
    x = torch.stack([unet_transforms(x) for x, _ in dataset])
    y = torch.tensor([y for _, y in dataset])
    train_dataset = TensorDataset(x, y)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.unet_batch_size, shuffle=True)

    weight_dtype = torch.float16
    accelerator = Accelerator(mixed_precision="fp16")
    unet = UNet2DModel(
        sample_size=32,
        in_channels=1,
        out_channels=1,
        block_out_channels=(32,64,128,256),
        norm_num_groups=8
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=500)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    for _ in tqdm(range(opt.unet_epochs), desc=f"UNet Class {cls} | Epochs"):
        for x, _ in train_dataloader:
            with accelerator.accumulate(unet):
                x = x.to(dtype=weight_dtype)
                noise = torch.randn_like(x, dtype=weight_dtype) # sample noise to add to clean images
                bsz = x.size(0)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
                timesteps = timesteps.long()
                noisy_images = noise_scheduler.add_noise(x, noise, timesteps) # add noise to images

                model_pred = unet(
                    noisy_images,
                    timesteps,
                    return_dict=False
                )[0] # unet noise prediction

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

    unet = accelerator.unwrap_model(unet)
    torch.save(unet.state_dict(), f"vgr-unet/perm_mnist/unet_task_{task_id}_class_{cls}.pt")
    accelerator.free_memory()
    del unet, optimizer, train_dataloader