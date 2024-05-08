from EntropyCodec import *

import torch
import numpy as np
from visualization_and_metrics import *

def EntropyEncoder(enc_img, size_z, size_h, size_w):
    temp = enc_img.astype(np.uint8).copy()

    maxbinsize = size_h * size_w * size_z
    bitstream = np.zeros(maxbinsize, np.uint8)
    StreamSize = np.zeros(1, np.int32)
    HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, StreamSize)
    return bitstream[: StreamSize[0]]


def EntropyDecoder(bitstream, size_z, size_h, size_w):
    decoded_data = np.zeros((size_z, size_h, size_w), np.uint8)
    FrameOffset = np.zeros(1, np.int32)
    HiddenLayersDecoder(decoded_data, size_w, size_h, size_z, bitstream, FrameOffset)
    return decoded_data


def process_images(test_loader, model, device, b, w, h):
    imgs_encoded = []
    imgs_decoded = []

    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            encoded_images = model.encoder(test_batch)
            decoded_images = model.decoder(encoded_images)

            imgs_encoded.append(encoded_images.cpu().detach())
            imgs_decoded.append(decoded_images.cpu().detach())

    imgs_encoded = torch.vstack(imgs_encoded)
    imgs_decoded = torch.vstack(imgs_decoded)


    max_encoded_imgs = imgs_encoded.amax(dim=1, keepdim=True)
    # Normalize and quantize
    norm_imgs_encoded = imgs_encoded / max_encoded_imgs
    quantized_imgs_encoded = (torch.clip(norm_imgs_encoded, 0, 0.9999999) * pow(2, b)).to(
        torch.int32
    )
    quantized_imgs_encoded = quantized_imgs_encoded.numpy()

    # Encode and decode using entropy coding
    quantized_imgs_decoded = []
    bpp = []

    for i in range(quantized_imgs_encoded.shape[0]):
        size_z, size_h, size_w = quantized_imgs_encoded[i].shape
        encoded_bits = EntropyEncoder(quantized_imgs_encoded[i], size_z, size_h, size_w)
        byte_size = len(encoded_bits)
        bpp.append(byte_size * 8 / (w * h))
        quantized_imgs_decoded.append(EntropyDecoder(encoded_bits, size_z, size_h, size_w))
    quantized_imgs_decoded = torch.tensor(np.array(quantized_imgs_decoded, dtype=np.uint8))

    shift = 1.0 / pow(2, b + 1)
    dequantized_imgs_decoded = (quantized_imgs_decoded.to(torch.float32) / pow(2, b)) + shift
    dequantized_denorm_imgs_decoded = dequantized_imgs_decoded * max_encoded_imgs

    imgsQ_decoded = []

    with torch.no_grad():
        for deq_img in dequantized_denorm_imgs_decoded:
            deq_img = deq_img.to(device)
            decoded_imgQ = model.decoder(deq_img)

            imgsQ_decoded.append(decoded_imgQ.cpu().detach())

    imgsQ_decoded = torch.stack(imgsQ_decoded)

    assert imgsQ_decoded.shape == imgs_decoded.shape
    assert imgsQ_decoded.shape[0] == len(bpp)

    return imgs_decoded, imgsQ_decoded, bpp