import wget, os, gzip, pickle, re, sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import wandb
import random
import torch, os, time, math, gzip
from tqdm import tqdm
import torch.distributions as dist
import argparse

from torch.utils.tensorboard import SummaryWriter

import numpy as np

### DATA PREPROCESSING ###

wandb.login(key='694fb34144778f8ff751adfb317024489e1a077e')
def enwik8(path=None, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    

    if path is None:
        path = here('/home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/data/enwik8.gz')

    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def enwik8_bytes(path=None, split=(90, 5, 5)):
    """
    Load the enwik8 dataset from the Hutter challenge as a python list of bytes

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """


    if path is None:
        path = here('/home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/data/enwik8.gz')

    with gzip.open(path, 'r') if path.endswith('.gz') else open(path, 'rb') as file:
        all = file.read()

        split = tuple(s/sum(split) for s in split)
        split = tuple(int(s * len(all)) for s in split)

        train, val, test = all[:split[0]], all[split[0]:split[0]+split[1]], all[split[0]+split[1]:]

        return train, val, test


def enwik8_string(path=None, split=(90, 5, 5)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """


    if path is None:
        path = here('/home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/data/enwik8.gz')

    with gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r') as file:
        all = file.read()

        split = tuple(s/sum(split) for s in split)
        split = tuple(int(s * len(all)) for s in split)

        train, val, test = all[:split[0]], all[split[0]:split[0]+split[1]], all[split[0]+split[1]:]
        return train, val, test


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()


def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    return seed


def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shifted one position to the right, to provide as a
    target for the model.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs  = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if '__file__' not in locals():
        __file__ = os.getcwd()

    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))


def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

def tic():
    tics.append(time.time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()

def slice_diag(matrix, l, dv=None):
    """
    Take a batch of attention matrices for relative position encodings and slice out the relevant attentions. These
    are the length l sequences starting at the diagonal

    :param matrix:
    :return:
    """
    if dv is None:
        dv = d(matrix)

    h, w = matrix.size(-2), matrix.size(-1)

    assert w == 2 * l -1, f'(h, w)= {(h, w)}, l={l}'

    rest = matrix.size()[:-2]

    matrix = matrix.view(-1, h, w)
    b, h, w = matrix.size()

    result = matrix.view(b, -1)
    result = torch.cat([result, torch.zeros(b, l, device=dv)], dim=1)
    assert result.size() == (b, 2 * l * l), f'result.size() {result.size()}'

    result = result.view(b, l, 2*l)
    result = result[:, :, :l]

    result = result.view(*rest, h, l)
    return result

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)

def compute_compression(model, data, context, batch_size, verbose=False,
                        tbw:SummaryWriter=None, tok=None, skip=0):


    """
    Compute the _compression_ of a dataset under a model. That is, given a model, in how many bits could we represent
    the dataset. This requires us to turn a given probability distribution into a code for the outcomes.

    See [this video](https://youtu.be/mSneVjDvzNQ) for an explanation.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the  data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []
    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.

    target_indices = []
    i, ic = 0, 0

    for current in tqdm.trange(skip, data.size(0)) if verbose else range(skip, data.size(0)):

        # `current` is the character which we will ultimately predict

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        # if tok is not None:
        #     print(instance[:-1], tok.decode(instance[:-1]))
        #     print(instance[-1:], tok.decode(instance[-1:]))

        target_indices.append(instance.size(0) - 2) # index of the last element of the input to the model

        if instance.size(0) < context + 1:
            assert skip < context # We shouldn't get here if we skip the first `context` characters

            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or current == data.size(0) - 1:
            # batch is full or we are at the last instance, run it through the model

            b = len(batch)

            ti = torch.tensor(target_indices) + 1
            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[torch.arange(b), ti]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=d()), target_indices, target]
            log2probs = lnprobs / LOGE2
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            if tbw is not None:
                for j, lp in enumerate(log2probs):
                    i += 1
                    tbw.add_scalar('compression/bits-per-token', -lp, i)

                    if tok is not None:
                        nc = len(tok.decode(target[j]))
                        ic += nc
                        tbw.add_scalar('compression/bits-per-byte', -lp/nc, ic)

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilities) to the running total
            batch, target_indices = [], []  # clear the buffer

    if isinstance(bits, torch.Tensor):
        bits = bits.item()

    return bits # total nr of bits used

def estimate_compression(model, data, nsamples, context, batch_size, verbose=False, model_produces_logits=False):
    """
    Estimates the compression by sampling random subsequences instead of predicting all characters.

    NB: This doesn't work for GPT-2 style models with super-character tokenization, since the tokens and number of
    characters are mismatched.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []

    # indices of target characters in the data
    gtargets = random.sample(range(data.size(0)), k=nsamples)

    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.
    target_indices = []

    for i, current in enumerate(tqdm.tqdm(gtargets) if verbose else gtargets):
        # current is the character to be predicted

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        target_indices.append(instance.size(0) - 2) # index of the last element of the context

        if instance.size(0) < context + 1:
            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or i == len(gtargets) - 1:
            # batch is full, or we are at the last instance, run it through the model

            b = len(batch)

            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[:, -1]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

                if model_produces_logits:
                    output = F.log_softmax(output, dim=-1)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=d()), target_indices, target]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch, target_indices = [], []  # clear the buffer

    return bits.item() / nsamples # total nr of bits used

def batcher(data, seq_len, batch_size, num_batches):
    x_batches, y_batches = [],[]
    
    for i in range(num_batches):
        start_idx = i * batch_size * (seq_len + 1)
        x_batch,y_batch = [],[]
        for _ in range(batch_size):
            x_batch.append(data[start_idx:start_idx + seq_len])
            y_batch.append(data[start_idx + 1: start_idx + seq_len + 1])

            start_idx += seq_len + 1
        
        x_batch = torch.cat([x[None,:] for x in x_batch], dim=0).to(torch.long)
        y_batch = torch.cat([y[None,:] for y in y_batch], dim=0).to(torch.long)

        x_batches.append(x_batch)
        y_batches.append(y_batch)

    return x_batches, y_batches


### DATA PREPROCESSING COMPLETE ###

### TRANSFORMER MODEL ###

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads if input_dim % num_heads == 0 else (input_dim // num_heads) + 1

        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.W_o = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        

        #Linear projections of Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split tensors into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        
        # Concatenate
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, dropout=0.1):
        super().__init__()

        self.attention = SelfAttention(k, heads, dropout)
        self.layer_norm = nn.LayerNorm(k)
        # Feedforward layer
        self.linear1 = nn.Linear(k, 4 * k)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * k, k)
        # Layer normalization after feedforward
        self.layer_norm_ff = nn.LayerNorm(k)
    

    def forward(self, x):
        x = self.layer_norm(x)
        attention_output = self.attention(x)
        x = x + attention_output  # Applying residual connection
        x = self.layer_norm_ff(x)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        output = x + ff_output  # Applying another residual connection
        return output
    
class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        # Embeddings and transformer blocks setup
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        self.tblocks = nn.Sequential(
            *[TransformerBlock(k=k, heads=heads) for _ in range(depth)]
        )
        self.output_layer = nn.Linear(k, num_classes)  

    def forward(self, x):
        tokens = self.token_emb(x)  # [batch_size, seq_length, k]
        positions = torch.arange(tokens.size(1), device=x.device).unsqueeze(0)
        x = tokens + self.pos_emb(positions)  # Add position embeddings

        x = self.tblocks(x)  # Pass through Transformer blocks

        x = self.output_layer(x)  # [batch_size, seq_length, num_classes]
        return F.log_softmax(x, dim=1)  # No pooling here, we need raw outputs per timestep
    
 
    
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
                # TODO detokenise output to make it human readable (GPT)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100
        return avg_loss, accuracy

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


# detokenize the model's prediction
def main(args):
    tics = []
    # Define all your settings
    config = {  # TODO: remove redundant settings >> maybe use args to dynamically set them
        "learning_rate": args.lr,
        "architecture": "Transformer",
        "dataset": "einwik8",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sequence_length": 256,
        "print_interval": 100,
        "validation_interval": 100,
        "model_params": {
            "k": 512,
            "heads": 8,
            "depth": 6,
            "seq_length": 256,
            "num_tokens": 256,
            "num_classes": 256
        },
        "optimizer_params": {
            "lr": 0.0001
        },
    }


    wandb.init(  # set the wandb project where this run will be logged
        project="thesis-project",
        config=config
    )

    train_data, val_data, test_data = enwik8()

    test_batches_x, test_batches_y = batcher(train_data[:100], seq_len=5, batch_size=3, num_batches=2)
    print(f'batch size x: {len(test_batches_x)}, y: {len(test_batches_y)}')
    print(f'INPUT batches:\n{test_batches_x}\nTARGET batches: \n{test_batches_y}')

    #model = Transformer(k=256, heads=4, depth=3, seq_length=256, num_tokens=256, num_classes=256)
    # TODO ask gpt how to pass model params to model directly
    # model = Transformer(depth=3, **config)

    # Parameters and setup
    model.to(d())
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), )

    model = Transformer(**config["model_params"])
    model.to(d())
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"]) #lr=args.lr

    # Early stopping and checkpointing setup
    patience = 20  # How many intervals to wait before stopping
    best_val_loss = float('inf')
    counter = 0  # Counter for early stopping

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        inputs, targets = sample_batch(train_data, length=config["sequence_length"], batch_size=config["batch_size"])
        inputs, targets = inputs.to(d()), targets.to(d())

        # Forward pass
        optimizer.zero_grad()
        output = model(inputs)
        output = output.view(-1, 256)
        targets = targets.view(-1)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % config["print_interval"] == 0:
            print(f'Epoch {epoch + 1}: Training Loss = {loss.item()}')

        wandb.log({"training_loss": loss.item()})

        if (epoch + 1) % config["validation_interval"] == 0:
            val_loss, val_accuracy = validate(model, val_data, criterion, config["batch_size"])
            print(f'Epoch {epoch + 1}: Validation Loss = {val_loss}, Accuracy = {val_accuracy}%')
            wandb.log({"validation_loss": val_loss, "validation_accuracy": val_accuracy})

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                wandb.save('best_model.pth')  # Save the model to wandb
                print("Saved best model")
                counter = 0  # reset the counter
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)  # Note the change from int to float

    args = parser.parse_args()
    main(args)
