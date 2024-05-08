from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm


TEST_FOLDER = os.path.join('..', 'data', 'test')
CKPT_PATH = os.path.join('..', 'weights')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ImageDataset(Dataset):
    def __init__(self, foldername, augfoldername=None, transform=None):
        files_as_is = [os.path.join(foldername, f) for f in os.listdir(foldername) if f.endswith(('.png', '.jpg', '.jpeg'))]
        aug_files = []
        if not augfoldername is None:
            aug_files = [os.path.join(augfoldername, f) for f in os.listdir(augfoldername) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.file_paths = [*files_as_is, *aug_files]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img = Image.open(self.file_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def train(model, data, epochs=100, lr=1e-3, gamma = 0.97, step_size=1, device = DEVICE, ckpt_path=CKPT_PATH):
    losses = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma, verbose=True)
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        pbar = tqdm(data)
        cur_losses = []
        for x in pbar:
            x = x.to(device)
            opt.zero_grad()
            x_hat = model(x)
            loss = F.mse_loss(x, x_hat)
            #loss = F.mse_loss(x, x_hat) + model.vq_loss
            loss.backward()
            opt.step()

            pbar.set_postfix({'loss': loss.item()})
            cur_losses.append(loss.item())
        losses.append(np.array(cur_losses).mean())
        scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), ckpt_path)
    plt.plot(losses)
    plt.show()