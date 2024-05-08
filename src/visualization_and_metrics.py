import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt

to_pil_transform = transforms.ToPILImage()


def PSNR_RGB(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    mse = torch.mean(torch.square(y_pred - y_true))
    if mse == 0:
        return float('inf')  # consistent handling of zero MSE
    return (10 * torch.log10(max_pixel ** 2 / mse)).item()

def JPEGRDSingleImage(torch_img, TargetBPP):
    image = to_pil_transform(torch_img)

    width, height = image.size
    realbpp = 0
    realpsnr = 0
    realQ = 0
    for Q in range(101):
        image.save("test.jpeg", "JPEG", quality=Q)
        image_dec = Image.open("test.jpeg")
        bytesize = os.path.getsize("test.jpeg")
        bpp = bytesize * 8 / (width * height)
        psnr = PSNR_RGB(np.array(image), np.array(image_dec))
        if abs(realbpp - TargetBPP) > abs(bpp - TargetBPP):
            realbpp = bpp
            realpsnr = psnr
            realQ = Q
    return image, realQ, realbpp, realpsnr

def display_images_and_save_pdf(test_dataset, imgs_decoded, imgsQ_decoded, bpp, filepath, NumImagesToShow=None):
    if NumImagesToShow is None:
        NumImagesToShow = len(test_dataset)
    cols = NumImagesToShow
    rows = 4

    plt.figure(figsize=(2 * cols, 2 * rows))

    for i in range(NumImagesToShow):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(to_pil_transform(test_dataset[i]), interpolation="nearest")
        plt.title("", fontsize=10)
        plt.axis('off')
    
    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgs_decoded[i])
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(to_pil_transform(imgs_decoded[i]), interpolation="nearest")
        plt.title(f"{psnr:.2f}", fontsize=10)
        plt.axis('off')
    
    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgsQ_decoded[i])
        plt.subplot(rows, cols, 2 * cols + i + 1)
        plt.imshow(to_pil_transform(imgsQ_decoded[i]), interpolation="nearest")
        plt.title(f"{psnr:.2f} {bpp[i]:.2f}", fontsize=10)
        plt.axis('off')
    

    for i in range(NumImagesToShow):
        jpeg_img, JPEGQP, JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(test_dataset[i], bpp[i])
        plt.subplot(rows, cols, 3 * cols + i + 1)
        plt.imshow(jpeg_img, interpolation="nearest")
        plt.title(f"{JPEGrealpsnr:.2f} {JPEGrealbpp:.2f}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filepath, format='pdf')
    plt.show()
