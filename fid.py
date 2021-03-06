import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm

class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensor, transform=None):
        self.tensor = tensor
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensor[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return self.tensor.size(0)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def load_model(device):
    m = models.inception_v3(pretrained=True, aux_logits=False).to(device)
    m.fc = Identity()
    m.eval()

    return m

def get_activations(model, batch_size, test_loader, fake_loader):
    
    assert len(test_loader) == len(fake_loader), "Loaders lengths must match"
    l = len(test_loader)
    
    start_idx = 0
    real_pred = np.empty((batch_size * l, 2048))
    fake_pred = np.empty((batch_size * l, 2048))
    
    for i, (test_batch, fake_batch) in tqdm(enumerate(zip(test_loader, fake_loader))):
        test_batch = test_batch[0]
        if test_batch.size(1) != 3:
            test_batch = test_batch.expand(-1, 3, -1, -1)
        if fake_batch.size(1) != 3:
            fake_batch = fake_batch.expand(-1, 3, -1, -1)
        
        with torch.no_grad():
            real_stats = model(test_batch).cpu().numpy()
            fake_stats = model(fake_batch).cpu().numpy()

        real_pred[start_idx:start_idx + batch_size] = real_stats
        fake_pred[start_idx:start_idx + batch_size] = fake_stats
        
        start_idx += batch_size
    
    return real_pred, fake_pred


def compute_fid(real_pred, fake_pred):
    
    mu_real, sigma_real = real_pred.mean(axis=0), np.cov(real_pred, rowvar=False)
    mu_fake, sigma_fake = fake_pred.mean(axis=0), np.cov(fake_pred, rowvar=False)

    ssdif = np.sum((mu_real - mu_fake)**2.)
    covmean = sqrtm(sigma_real.dot(sigma_fake))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return np.trace(sigma_real + sigma_fake - 2.*covmean)