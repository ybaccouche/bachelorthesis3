import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.init as init
from itertools import product
import wandb
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
import psutil  # For resource utilization
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

class Perceiver(nn.Module):
    def __init__(self, vocab_size, emb_size, max_seq_length, latent_dim, num_latents, nblocks, nheads, dropout, ff_hidden):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, max_seq_length)
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        self.perceiver_blocks = nn.ModuleList(
            [PerceiverBlock(emb_size, latent_dim, nheads, dropout, ff_hidden) for _ in range(nblocks)]
        )
        self.to_logits = nn.Linear(latent_dim, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding_layer(x)
        x = self.pos_encoder(x)
        latent = self.latents.expand(batch_size, -1, -1)
        for perceiver_block in self.perceiver_blocks:
            latent = perceiver_block(x, latent)
        y = self.to_logits(latent)
        lg_probs = F.log_softmax(y, dim=-1)
        return lg_probs

wandb.login(key='694fb34144778f8ff751adfb317024489e1a077e')
# NEW W&B RUN
wandb.init(  # set the wandb project where this run will be logged
    project="thesis-project"
)

def kaiming_init_weights(model):
    if isinstance(model, nn.Linear):
        init.kaiming_uniform_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.Embedding):
        init.uniform_(model.weight, -0.1, 0.1)
    elif isinstance(model, nn.LayerNorm):
        init.constant_(model.bias, 0)
        init.constant_(model.weight, 1.0)

def calculate_perplexity(loss):
    return torch.exp(loss)

def calculate_bleu(outputs, targets):
    # Assuming outputs and targets are lists of sentences (strings)
    bleu_scores = [sentence_bleu([tgt.split()], out.split()) for out, tgt in zip(outputs, targets)]
    return sum(bleu_scores) / len(bleu_scores)

def log_resource_utilization(step):
    # Log CPU and memory utilization
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent


    wandb.log({
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        '_step': step
    })
        
# HYPERPARAMS
parser = argparse.ArgumentParser(description='Perceiver Model') 
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--seq_length', type=int, default=256, help='number of characters per training sequence')
parser.add_argument('--batch_size', type=int, default=32, help='number of text sequences per batch')
parser.add_argument('--input_dim', type=int, default=128, help='input dimension')
parser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')
parser.add_argument('--vocab_size', type=int, default=241, help='vocabulary size')
parser.add_argument('--nblocks', type=int, default=12, help='number of perceiver blocks')
parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--ff_hidden', type=int, default=512, help='size of feedforward hidden layer in perceiver blocks')
parser.add_argument('--num_latents', type=int, default=256, help='number of latents')
args = parser.parse_args()

learning_rate = args.learning_rate
seq_length = args.seq_length
batch_size = args.batch_size
input_dim = args.input_dim
latent_dim = args.latent_dim
vocab_size = args.vocab_size
nblocks = args.nblocks
nheads = args.nheads
dropout = args.dropout
ff_hidden = args.ff_hidden
num_latents = args.num_latents

param_grid = {
    'learning_rate': [0.0001, 0.001],
    'seq_length': [128, 256],
    'batch_size': [32, 128],
    'input_dim': [128, 256],
    'latent_dim': [128, 256],
    'nblocks': [6, 12],
    'nheads': [4, 8],
    'dropout': [0.2, 0.3],
    'ff_hidden': [512, 1024],
    'num_latents': [128, 256],
}

# Generate all combinations of hyperparameters
param_combinations = list(product(*param_grid.values()))

best_hyperparams = None
best_val_loss = float('inf')

# TRAINING
# ~48 hours for 120k batches (~23 mins per 1000 batches on TitanX)

#learning_rate = 0.0001

#seq_length = 256  # no. of chars per training sequence
#batch_size = 32  # no. of text sequences per batch
num_batches = 10000  # no. of batches to train on
log_interval = 500  # num batches b/w logging training progress

#input_dim = 128
#latent_dim = 128
#vocab_size = 241  # data chars 9 - 240
#nblocks, nheads = 12, 8  # no. of perceiver blocks, and attn heads
#dropout = 0.1  # dropout probability
#ff_hidden = 4 * latent_dim  # size of feedforward hidden layer in perceiver blocks
#num_latents = 256  # number of latents

# VALIDATION
sampling_temp = 0.8  # for scaling predicted probs for next char
sample_length = 600  # length of text to sample/generate
nchars_compression = 10000  # num of chars to predict for estimating compression

sample_interval = 10000  # num batches b/w sampling while training


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
    model.eval()
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
            total_bits += -log_2_probs.sum().item()

            buffer, ctx_idxs, target_chars = [], [], []

    bpc = total_bits / nchars_compression

    print(f'BITS PER CHAR after {batch_num} batches: {bpc:.2f}\n')
    wandb.log({'bits per character': bpc})


# VALIDATION LOSS
def estimate_val_loss(model):
    inps, targs = batcher(val_data, seq_length, batch_size)
    val_out = model(inps)
    val_out = val_out.view(-1, val_out.size(-1))
    targs = targs.view(-1)
    return F.nll_loss(F.log_softmax(val_out, dim=-1), targs, reduction='mean')


for params in param_combinations:
    hyperparams = dict(zip(param_grid.keys(), params))
    print(f"Training with hyperparameters: {hyperparams}")

    # Set hyperparameters
    learning_rate = hyperparams['learning_rate']
    seq_length = hyperparams['seq_length']
    batch_size = hyperparams['batch_size']
    input_dim = hyperparams['input_dim']
    latent_dim = hyperparams['latent_dim']
    nblocks = hyperparams['nblocks']
    nheads = hyperparams['nheads']
    dropout = hyperparams['dropout']
    ff_hidden = hyperparams['ff_hidden']
    num_latents = hyperparams['num_latents']

    # Initialize model, optimizer, and scheduler
    model = Perceiver(vocab_size, input_dim, seq_length, latent_dim, num_latents, nblocks, nheads, dropout, ff_hidden).to(device)
    model.apply(kaiming_init_weights)
    opt = AdamW(params=model.parameters(), lr=learning_rate, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=num_batches, eta_min=learning_rate / 1000)

    # Early stopping parameters
    patience = 1000
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for i in range(num_batches):
        if early_stop:
            print("Early stopping")
            break

        model.train()
        opt.zero_grad()

        inputs, targets = batcher(train_data, seq_length, batch_size)

        outputs = model(inputs)
        loss = F.nll_loss(outputs.transpose(2, 1), targets, reduction='mean')

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sch.step()

        # Calculate gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_grad_norm = p.grad.data.norm(2)
                total_grad_norm += param_grad_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        if i == 0 or (i + 1) % log_interval == 0:
            print(f'(batch {i+1:6d}) train loss: {loss.item():.4f}')
            model.eval()
            with torch.no_grad():
                val_loss = estimate_val_loss(model)
                train_perplexity = calculate_perplexity(loss)
                val_perplexity = calculate_perplexity(val_loss)

                # Dummy predictions and targets for BLEU score, replace with actual values
                val_outputs = ["predicted sentence"] * len(targets)
                val_targets = ["actual sentence"] * len(targets)
                bleu_score = calculate_bleu(val_outputs, val_targets)

                wandb.log({
                    'train loss': loss.item(),
                    'validation loss': val_loss.item(),
                    'train perplexity': train_perplexity.item(),
                    'validation perplexity': val_perplexity.item(),
                    'bleu score': bleu_score,
                    'gradient norm': total_grad_norm,
                    '_step': i + 1
                })

                # Log resource utilization
                log_resource_utilization(i + 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 'Perceiver_best.pt')
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    early_stop = True
                    print("Early stopping triggered")
                    break

        if i == 0 or (i + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                sampler(model)
                estimate_compression(model, i + 1)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_hyperparams = hyperparams

print(f'Best hyperparameters: {best_hyperparams}')
print('training complete')
wandb.finish()