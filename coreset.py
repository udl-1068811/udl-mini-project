import torch
import torch.utils.data as data
from copy import deepcopy
import numpy as np
import torch.optim
from tqdm import tqdm
from random import shuffle
from torch.utils.data import Subset, ConcatDataset, DataLoader

class Coreset():
    """
    Base class for the the coreset.  This version of the class has no
    coreset but subclasses will replace the select method.
    """

    def __init__(self, size=0):
        self.size = size
        self.coreset = None
        self.coreset_task_ids = None

    def coreset_train(
        self,
        model,
        optim,
        lr,
        tasks,
        epochs,
        device,
    ):
        """
        Train model on the coreset and return the trained model.
        """

        print("Training on coreset")

        if self.coreset is None:
            return model

        model = deepcopy(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(optim.state_dict())

        # if tasks is an integer, turn it into a singleton.
        if isinstance(tasks, int):
            tasks = [tasks]

        # create dict of train_loaders
        train_loaders = {}
        for task_id in tasks:
            coreset_task_indices = torch.where(self.coreset_task_ids == task_id)[0]
            coreset_task_subset = Subset(self.coreset, coreset_task_indices)
            train_loaders[task_id] = DataLoader(coreset_task_subset, 256)

        for _ in tqdm(range(epochs), desc='Epochs'):
            # Randomize order of training tasks
            shuffle(tasks)
            for task_id in tasks:

                for x, y in train_loaders[task_id]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = model.vcl_loss(x, y, 0, len(self.coreset), 10)
                    loss.backward()
                    optimizer.step()

        return model


class RandomCoreset(Coreset):

    def __init__(self, size):
        super().__init__(size)

    def select(self, data: data.Dataset, task_id: int):
        # Generate random indices for coreset and non-coreset samples
        num_samples = len(data)
        coreset_indices = torch.randperm(num_samples)[:self.size]
        non_coreset_indices = torch.tensor([i for i in range(num_samples) if i not in coreset_indices])

        # Create coreset and non-coreset subsets
        coreset_subset = Subset(data, coreset_indices)
        non_coreset_subset = Subset(data, non_coreset_indices)

        # Update coreset and task_ids
        if self.coreset is None:
            self.coreset = coreset_subset
            self.coreset_task_ids = torch.full((len(coreset_subset),), task_id)
        else:
            self.coreset = ConcatDataset([self.coreset, coreset_subset])
            self.coreset_task_ids = torch.cat((self.coreset_task_ids, torch.full((len(coreset_subset),), task_id)))

        return non_coreset_subset
    
class KCenterCoreset(Coreset):

    def __init__(self, size):
        super().__init__(size)
    
    def select(self, data: data.Dataset, task_id: int):
        num_samples = len(data)
        dists = torch.full((num_samples,), float('inf'))
        current_id = 0
        x = torch.stack([x for x, _ in data])
        dists = self.update_distance(dists, x, current_id)
        idx = [current_id]

        for _ in range(1, self.size):
            current_id = torch.argmax(dists)
            dists = self.update_distance(dists, x, current_id)
            idx.append(current_id)

        coreset_indices = torch.tensor(idx)
        non_coreset_indices = torch.tensor([i for i in range(num_samples) if i not in coreset_indices])

        coreset_subset = Subset(data, coreset_indices)
        non_coreset_subset = Subset(data, non_coreset_indices)

        if self.coreset is None:
            self.coreset = coreset_subset
            self.coreset_task_ids = torch.full((len(coreset_subset),), task_id)
        else:
            self.coreset = ConcatDataset([self.coreset, coreset_subset])
            self.coreset_task_ids = torch.cat((self.coreset_task_ids, torch.full((len(coreset_subset),), task_id)))
        
        return non_coreset_subset

    def update_distance(self, dists, data, current_id):
        for i, v in enumerate(data):
            current_dist = torch.norm(v - data[current_id])
            dists[i] = min(current_dist.item(), dists[i].item())
        return dists