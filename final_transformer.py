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
import argparse

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

n_train, n_val, n_test = 90000000, 5000000, 5000000 # 90m, 5m, 5m

train_data, val_data, test_data = data_loader(data_path, n_train, n_val, n_test)

wandb.login(key='694fb34144778f8ff751adfb317024489e1a077e')
wandb.init(  # set the wandb project where this run will be logged
        project="thesis-project"
    )

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

# MULTI-HEAD ATTENTION
class SelfAttention(nn.Module):
    def __init__(self, emb_size, nheads, mask=True):
        super().__init__()
        self.emb_size, self.heads = emb_size, nheads
        self.head_dim = emb_size // nheads
        self.mask = mask
        self.qkv_transform = nn.Linear(emb_size, 3 * emb_size, bias=False)
        self.linear_transform = nn.Linear(emb_size, emb_size)

    def _attention(self, qkv, batch_size, seq_length):
        qkv = qkv.reshape(batch_size, seq_length, self.heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        queries, keys, values = qkv.chunk(3, dim=-1)
        weights = torch.matmul(queries, keys.transpose(-2, -1))
        weights = weights / queries.size(-1) ** (1 / 2)
        if self.mask:
            weights = weights.masked_fill(torch.tril(torch.ones_like(weights)) == 0.0, value=float('-inf'))
        weights = F.softmax(weights, dim=-1)
        out = torch.matmul(weights, values).permute(0, 2, 1, 3)
        return out.reshape(batch_size, seq_length, self.emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_transform(x)
        attn_out = self._attention(qkv, batch_size, seq_len)
        lin_transf = self.linear_transform(attn_out)
        return lin_transf

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, nheads, dropout, ff_hidden, mask):
        super().__init__()
        self.attention = SelfAttention(emb_size, nheads, mask=mask)
        self.emb_dim, self.heads = emb_size, nheads
        self.layer_norm_1 = nn.LayerNorm(emb_size)
        self.layer_norm_2 = nn.LayerNorm(emb_size)
        self.drop = nn.Dropout(p=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size, ff_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ff_hidden, emb_size)
        )

    def forward(self, x):
        att = self.attention(x)
        n1 = self.layer_norm_1(self.drop(att) + x)
        ff = self.feedforward(n1)
        n2 = self.layer_norm_2(self.drop(ff) + n1)
        return n2

# POSITIONAL ENCODINGS
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_seq_length):
        super().__init__()
        self.max_length = max_seq_length
        pos_enc = torch.zeros(self.max_length, emb_size)
        position = torch.arange(0, self.max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc, persistent=False)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)]
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_size, max_seq_length, nblocks, nheads, dropout, ff_hidden, mask=True):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, max_seq_length)
        self.transformer_blocks = nn.Sequential(*nn.ModuleList(
            [TransformerBlock(emb_size, nheads, dropout, ff_hidden, mask) for _ in range(nblocks)]))
        self.to_logits = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.pos_encoder(x)
        x = self.transformer_blocks(x)
        y = self.to_logits(x)
        lg_probs = F.log_softmax(y, dim=2)
        return lg_probs

def sampler(model, val_data, seq_length, sample_length, sampling_temp):
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

def estimate_compression(model, val_data, seq_length, batch_size, nchars_compression, batch_num):
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

def estimate_val_loss(model, val_data, seq_length, batch_size):
    inps, targs = batcher(val_data, seq_length, batch_size)
    val_out = model(inps)
    return F.nll_loss(val_out.transpose(2, 1), targs, reduction='mean')

def main(args):
    # HYPERPARAMS
    learning_rate = args.lr
    seq_length = 256  # no. of chars per training sequence
    batch_size = args.batch_size  # no. of text sequences per batch
    num_batches = args.num_batches  # no. of batches to train on 
    log_interval = 100  # num batches b/w logging training progress
    embed_size = 128
    vocab_size = 241  # data chars 9 - 240
    nblocks, nheads = args.nblocks, args.nheads  # no. of transformer blocks, and attn heads
    dropout = 0.1  # dropout probability
    ff_hidden = 4 * embed_size  # size of feedforward hidden layer in transf blocks
    mask = True  # whether to apply causal masking
    sampling_temp = 0.8  # for scaling predicted probs for next char
    sample_length = 600  # length of text to sample/generate
    nchars_compression = 10000  # num of chars to predict for estimating compression
    sample_interval = 500  # num batches b/w sampling while training

    # TRAINING
    model = Transformer(vocab_size, embed_size, seq_length, nblocks, nheads, dropout, ff_hidden, mask).to(device)
    opt = Adam(params=model.parameters(), lr=learning_rate)
    sch = CosineAnnealingLR(opt, T_max=num_batches, eta_min=learning_rate / 1000)  # learning rate scheduler

    best_val_loss = float('inf')

    for i in range(num_batches):
        model.train()
        opt.zero_grad()

        inputs, targets = batcher(train_data, seq_length, batch_size)
        outputs = model(inputs)
        loss = F.nll_loss(outputs.transpose(2, 1), targets, reduction='mean')
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sch.step()

        if i == 0 or (i + 1) % log_interval == 0:
            print(f'(batch {i+1:6d}) train loss: {loss.item():.4f}')
            model.eval()
            with torch.no_grad():
                val_loss = estimate_val_loss(model, val_data, seq_length, batch_size)
                wandb.log({'train loss': loss.item(), 'validation loss': val_loss.item(), '_step': i + 1})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'Transformer_longrun.pt')

        if i == 0 or (i + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                sampler(model, val_data, seq_length, sample_length, sampling_temp)
                estimate_compression(model, val_data, seq_length, batch_size, nchars_compression, i + 1)

    print('training complete')
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)  
    parser.add_argument('--nblocks', type=int, default=8) 
    parser.add_argument('--nheads', type=int, default=12) 
    args = parser.parse_args()
    main(args)