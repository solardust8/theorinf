import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.ResAE import *
from src.visualization_and_metrics import *
from src.Entropy_codec import *
from src.training import *

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src', 'codec_cpp'))
sys.path.append(os.path.join(os.getcwd(), 'src'))

TEST_PATH = os.path.join('.', 'data', 'test')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
WEIGHTS = os.path.join('.', 'weights', 'RAE_MISH_16x16x16latent_200epochs.ckpt')
WEIGHTS_base = os.path.join('.', 'weights', 'base_ae.pth')

test_dataset = ImageDataset(TEST_PATH)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


w, h = 128, 128
b = 3
bt = 3


MODEL = ResAE(activation=nn.Mish()).to(DEVICE)
#MODEL = BaseAutoEncoder().to(DEVICE)
MODEL.load_state_dict(torch.load(WEIGHTS, map_location=torch.device(DEVICE)))
MODEL.eval()

import faulthandler; faulthandler.enable()

imgs_decoded, imgsQ_decoded, bpp = process_images(test_loader, MODEL, DEVICE, b, w, h)
display_images_and_save_pdf(test_dataset, imgs_decoded, imgsQ_decoded, bpp, './misc/out1.jpg', NumImagesToShow=5)

