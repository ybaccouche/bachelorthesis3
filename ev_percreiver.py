import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import math, random, gzip

# use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Currently using:', device)


def data_loader(path, n_train, n_val, n_test):
    with gzip.open(path) as file:
        data = np.frombuffer(file.read(n_train + n_val + n_test), dtype=np.uint8)
        train, val, test = np.split(data, [n_train, n_train + n_val])
        return torch.from_numpy(train.copy()), torch.from_numpy(val.copy()), torch.from_numpy(test.copy())


# LOAD DATA (Wikipedia 'en' text, 100m chars)
data_path = '/home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/data/enwik8.gz'

n_train, n_val, n_test = 90000000, 5000000, 5000000  # 90m, 5m, 5m

train_data, val_data, test_data = data_loader(data_path, n_train, n_val, n_test)


# DECODER funcs
def token_to_string(token):
    return str(chr(max(32, token)))


def tokens_to_string(tokens):
    return ''.join(list(map(token_to_string, tokens)))


# BATCHING
def batcher(data, seq_length, batch_size):
    idxs = torch.randint(size=(batch_size,), high=data.size(0) - seq_length - 1)
    batch = torch.stack([data[i: i + seq_length + 1] for i in idxs], dim=0)
    return batch[:, :-1].to(torch.long).to(device), batch[:, 1:].to(torch.long).to(device)  # shift 1 char right for targets


# CROSS-ATTENTION LAYER
class CrossAttention(nn.Module):
    def __init__(self, latent_dim, input_dim, nheads):
        super().__init__()
        self.latent_dim = latent_dim
        self.nheads = nheads
        self.head_dim = latent_dim // nheads

        self.q_transform = nn.Linear(latent_dim, latent_dim, bias=False)
        self.kv_transform = nn.Linear(input_dim, 2 * latent_dim, bias=False)
        self.linear_transform = nn.Linear(latent_dim, latent_dim)

    def _attention(self, queries, keys, values):
        weights = torch.matmul(queries, keys.transpose(-2, -1))
        weights = weights / (self.head_dim ** 0.5)
        weights = F.softmax(weights, dim=-1)
        out = torch.matmul(weights, values)
        return out

    def forward(self, latents, inputs):
        batch_size, seq_len, _ = inputs.size()
        _, latent_len, _ = latents.size()

        queries = self.q_transform(latents).view(batch_size, latent_len, self.nheads, self.head_dim).transpose(1, 2)
        kv = self.kv_transform(inputs).view(batch_size, seq_len, self.nheads, 2 * self.head_dim).transpose(1, 2)
        keys, values = kv.chunk(2, dim=-1)

        attn_out = self._attention(queries, keys, values).transpose(1, 2).contiguous().view(batch_size, latent_len, self.latent_dim)
        return self.linear_transform(attn_out)


# SELF-ATTENTION LAYER (for latents)
class SelfAttention(nn.Module):
    def __init__(self, latent_dim, nheads):
        super().__init__()
        self.latent_dim = latent_dim
        self.nheads = nheads
        self.head_dim = latent_dim // nheads

        self.qkv_transform = nn.Linear(latent_dim, 3 * latent_dim, bias=False)
        self.linear_transform = nn.Linear(latent_dim, latent_dim)

    def _attention(self, queries, keys, values):
        weights = torch.matmul(queries, keys.transpose(-2, -1))
        weights = weights / (self.head_dim ** 0.5)
        weights = F.softmax(weights, dim=-1)
        out = torch.matmul(weights, values)
        return out

    def forward(self, latents):
        batch_size, seq_len, _ = latents.size()

        qkv = self.qkv_transform(latents).view(batch_size, seq_len, self.nheads, 3 * self.head_dim).transpose(1, 2)
        queries, keys, values = qkv.chunk(3, dim=-1)

        attn_out = self._attention(queries, keys, values).transpose(1, 2).contiguous().view(batch_size, seq_len, self.latent_dim)
        return self.linear_transform(attn_out)


class PerceiverBlock(nn.Module):
    def __init__(self, latent_dim, input_dim, nheads, dropout, ff_hidden):
        super().__init__()
        self.cross_attention = CrossAttention(latent_dim, input_dim, nheads)
        self.self_attention = SelfAttention(latent_dim, nheads)

        self.layer_norm_1 = nn.LayerNorm(latent_dim)
        self.layer_norm_2 = nn.LayerNorm(latent_dim)

        self.drop = nn.Dropout(p=dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(latent_dim, ff_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ff_hidden, latent_dim)
        )

    def forward(self, latents, inputs):
        cross_att = self.cross_attention(latents, inputs)
        latents = self.layer_norm_1(self.drop(cross_att) + latents)
        self_att = self.self_attention(latents)
        latents = self.layer_norm_2(self.drop(self_att) + latents)
        ff = self.feedforward(latents)
        return latents + self.drop(ff)


class Perceiver(nn.Module):
    def __init__(self, vocab_size, input_dim, latent_dim, num_latents, nblocks, nheads, dropout, ff_hidden):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, input_dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        self.perceiver_blocks = nn.ModuleList(
            [PerceiverBlock(latent_dim, input_dim, nheads, dropout, ff_hidden) for _ in range(nblocks)]
        )

        self.to_logits = nn.Linear(latent_dim, vocab_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        batch_size = x.size(0)
        latents = self.latents.expand(batch_size, -1, -1)

        for block in self.perceiver_blocks:
            latents = block(latents, x)

        y = self.to_logits(latents.mean(dim=1))  # Aggregate latents
        lg_probs = F.log_softmax(y, dim=1)
        return lg_probs


wandb.login(key='694fb34144778f8ff751adfb317024489e1a077e')
# NEW W&B RUN
wandb.init(  # set the wandb project where this run will be logged
    project="thesis-project"
)


# HYPERPARAMS

# TRAINING
# ~48 hours for 120k batches (~23 mins per 1000 batches on TitanX)

learning_rate = 0.001

seq_length = 256  # no. of chars per training sequence
batch_size = 100  # no. of text sequences per batch
num_batches = 5000  # no. of batches to train on
log_interval = 100  # num batches b/w logging training progress

input_dim = 128
latent_dim = 128
vocab_size = 241  # data chars 9 - 240
nblocks, nheads = 12, 8  # no. of perceiver blocks, and attn heads
dropout = 0.1  # dropout probability
ff_hidden = 4 * latent_dim  # size of feedforward hidden layer in perceiver blocks
num_latents = 256  # number of latents

# VALIDATION
sampling_temp = 0.8  # for scaling predicted probs for next char
sample_length = 600  # length of text to sample/generate
nchars_compression = 10000  # num of chars to predict for estimating compression

sample_interval = 100  # num batches b/w sampling while training


# SAMPLE a continuation for a random seed
def sampler(model):
    rand_idx = random.randint(0, val_data.size(0) - seq_length)
    seed = val_data[rand_idx: rand_idx + seq_length].to(torch.long).to(device)

    seed_text = tokens_to_string(seed)
    print(f'\n[SEED]: {seed_text[:128]}\n\t{seed_text[128:]}')

    for _ in range(sample_length):
        seed_tail = seed[-seq_length:]
        seed_batch = seed_tail[None, :].to(device)
        seed_out = model(seed_batch)

        last_log_probs = seed_out[0, -1, :]

        scaled_probs = F.softmax(last_log_probs / sampling_temp, dim=0)
        cat_dist = dist.Categorical(scaled_probs)
        char = cat_dist.sample()

        seed = torch.cat([seed, char[None]], dim=0)

    # generated/predicted text
    gen = tokens_to_string(seed[seq_length:])
    print(f'[PRED]: {gen[:125]}\n\t{gen[125:250]}\n\t{gen[250:375]}\n\t{gen[375:500]}\n\t{gen[500:]}\n')


# COMPRESSION
def estimate_compression(model, batch_num):
    rand_idxs = random.sample(range(1, len(val_data)), k=nchars_compression)

    buffer = []
    ctx_idxs = []
    target_chars = []
    total_bits = 0.0

    for idx in rand_idxs:
        start = max(0, idx - seq_length)
        ctx = val_data[start:idx].to(torch.long)
        ctx_idxs.append(len(ctx) - 1)
        target_chars.append(val_data[idx].to(torch.long))

        if len(ctx) < seq_length:
            pad = torch.zeros(size=(seq_length - len(ctx),), dtype=torch.long)
            ctx = torch.cat([ctx, pad], dim=0)

        buffer.append(ctx)

        if len(buffer) == 2 * batch_size or idx == rand_idxs[-1]:
            contexts = torch.stack(buffer, dim=0).to(device)
            preds = model(contexts)

            nat_log_probs = preds[torch.arange(len(buffer)), ctx_idxs, target_chars]
            log_2_probs = nat_log_probs / math.log(2.0)
            total_bits += -log_2_probs.sum()

            buffer, ctx_idxs, target_chars = [], [], []

    bpc = total_bits / nchars_compression

    print(f'BITS PER CHAR after {batch_num} batches: {bpc:.2f}\n')

    wandb.log({'bits per character': bpc})


# VALIDATION LOSS
def estimate_val_loss(model):
    inps, targs = batcher(val_data, seq_length, batch_size)
    val_out = model(inps)
    return F.nll_loss(val_out, targs, reduction='mean')


# TRAINING
model = Perceiver(vocab_size, input_dim, latent_dim, num_latents, nblocks, nheads, dropout, ff_hidden).to(device)

opt = Adam(params=model.parameters(), lr=learning_rate)
sch = CosineAnnealingLR(opt, T_max=num_batches, eta_min=learning_rate / 1000)  # learning rate scheduler

best_val_loss = float('inf')

for i in range(num_batches):
    model.train()
    opt.zero_grad()

    inputs, targets = batcher(train_data, seq_length, batch_size)

    outputs = model(inputs)
    loss = F.nll_loss(outputs, targets, reduction='mean')

    loss.backward()

    clip_grad_norm_(model.parameters(), max_norm=1.0)

    opt.step()
    sch.step()

    if i == 0 or (i + 1) % log_interval == 0:
        print(f'(batch {i + 1:6d}) train loss: {loss.item():.4f}')

        model.eval()

        with torch.no_grad():
            val_loss = estimate_val_loss(model)
            wandb.log({'train loss': loss.item(), 'validation loss': val_loss.item(), '_step': i + 1})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'Perceiver_longrun.pt')

    if i == 0 or (i + 1) % sample_interval == 0:
        model.eval()

        with torch.no_grad():
            sampler(model)
            estimate_compression(model, i + 1)

print('training complete')
wandb.finish()