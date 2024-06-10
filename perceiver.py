import wget, os, gzip, pickle, random, re, sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

class PerceiverAutoregressive(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_size, num_latent, num_layers, num_heads, num_iterations):
        super(PerceiverAutoregressive, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.latents = nn.Parameter(torch.randn(num_latent, latent_size))
        
        self.input_to_latent = nn.MultiheadAttention(embed_size, num_heads)
        self.latent_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(latent_size, num_heads, dim_feedforward=latent_size), num_layers
        )
        self.latent_to_output = nn.MultiheadAttention(latent_size, num_heads)
        self.fc_out = nn.Linear(latent_size, vocab_size)
        
        self.num_iterations = num_iterations

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_size)
        
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_latent, latent_size)
        
        for _ in range(self.num_iterations):
            # Cross-attention: inputs to latents
            latents, _ = self.input_to_latent(x.permute(1, 0, 2), latents.permute(1, 0, 2), latents.permute(1, 0, 2))
            latents = latents.permute(1, 0, 2)  # Shape: (batch_size, num_latent, latent_size)
            
            # Latent Transformer
            latents = self.latent_transformer(latents.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Cross-attention: latents to output
        output, _ = self.latent_to_output(latents.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))
        output = output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, latent_size)
        
        output = self.fc_out(output)  # Shape: (batch_size, seq_len, vocab_size)
        
        return output

# Example usage
vocab_size = 10000
embed_size = 512
latent_size = 512
num_latent = 128
num_layers = 6
num_heads = 8
num_iterations = 6

model = PerceiverAutoregressive(vocab_size, embed_size, latent_size, num_latent, num_layers, num_heads, num_iterations)

def validate(model, data, criterion, batch_size=32, sequence_length=256, num_batches=10):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients during validation
        for _ in range(num_batches):
            # Generate batches using sample_batch
            inputs, targets = sample_batch(data, length=sequence_length, batch_size=batch_size)
            inputs, targets = inputs.to(d()), targets.to(d())

            outputs = model(inputs)
            outputs = outputs.view(-1, 256)  # Flatten outputs for loss calculation
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # Aggregate the loss

            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
            print(f"Predicted: {predicted[:10]}, True: {targets[:10]}")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy