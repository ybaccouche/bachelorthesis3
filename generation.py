import wget, os, gzip, pickle, random, re, sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

seed_sequence = torch.randint(256, (256,)).to(d())

seed_sequence = torch.randint(256, (256,)).to(d())  # Seed sequence for text generation
max_context = 256  # Maximum context length
length = 256  # Length of generated text
temperature = 0.5  # Sampling temperature

sampled_sequence = sample_sequence(model, seed_sequence, max_context=max_context, length=length, temperature=temperature, verbose=True)

compression_bits_per_byte = estimate_compression(model, val_data, nsamples=1000, context=max_context, batch_size=32, verbose=True)