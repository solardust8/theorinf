import os
import numpy as np
import torch
import io

from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import copy

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
    init_img = copy.deepcopy(image)

    width, height = image.size
    realbpp = 0
    realpsnr = 0
    realQ = 0
    final_image = None

    for Q in range(101):
        img_bytes = io.BytesIO()
        image.save(img_bytes, "JPEG", quality=Q)
        img_bytes.seek(0)
        image_dec = Image.open(img_bytes)
        bytesize = len(img_bytes.getvalue())

        bpp = bytesize * 8 / (width * height)
        psnr = PSNR_RGB(np.array(init_img), np.array(image_dec))
        if abs(realbpp - TargetBPP) > abs(bpp - TargetBPP):
            realbpp = bpp
            realpsnr = psnr
            realQ = Q
            final_image = image_dec
            
    return final_image, realQ, realbpp, realpsnr

def display_images_and_save_pdf(test_dataset, imgs_decoded, imgsQ_decoded, bpp, filepath=None, NumImagesToShow=None):
    if NumImagesToShow is None:
        NumImagesToShow = len(test_dataset)
    cols = NumImagesToShow
    rows = 4

    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    psnr_decoded = []
    psnr_decoded_q = []
    psnr_jpeg = []

    for i in range(NumImagesToShow):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(to_pil_transform(test_dataset[i]), interpolation="nearest")
        plt.title("", fontsize=10)
        plt.axis('off')
    
    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgs_decoded[i])
        psnr_decoded.append(psnr)
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(to_pil_transform(imgs_decoded[i]), interpolation="nearest")
        plt.title(f"PSNR: {psnr:.2f}", fontsize=10)
        plt.axis('off')
    
    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgsQ_decoded[i])
        psnr_decoded_q.append(psnr)
        plt.subplot(rows, cols, 2 * cols + i + 1)
        plt.imshow(to_pil_transform(imgsQ_decoded[i]), interpolation="nearest")
        plt.title(f"PSNR: {psnr:.2f} | BPP: {bpp[i]:.2f}", fontsize=10)
        plt.axis('off')
    

    for i in range(NumImagesToShow):
        jpeg_img, JPEGQP, JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(test_dataset[i], bpp[i])
        psnr_jpeg.append(JPEGrealpsnr)
        plt.subplot(rows, cols, 3 * cols + i + 1)
        plt.imshow(jpeg_img, interpolation="nearest")
        plt.title(f"PSNR: {JPEGrealpsnr:.2f} | BPP: {JPEGrealbpp:.2f}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, format='pdf')
    return fig, np.mean(psnr_decoded), np.mean(psnr_decoded_q), np.mean(psnr_jpeg)

