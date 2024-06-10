# Parameters and setup
model = Transformer(k=256, heads=4, depth=3, seq_length=256, num_tokens=256, num_classes=256)
model.to(d())
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
total_batches = 100
batch_size = 32
sequence_length = 256
print_interval = 10
validation_interval = 10

# Early stopping and checkpointing setup
patience = 20  # How many intervals to wait before stopping
best_val_loss = float('inf')
counter = 0  # Counter for early stopping

# Progress bar
pbar = tqdm.tqdm(total=total_batches, desc="Training Batches")

# Training loop
for current_batch in range(total_batches):
    model.train()
    inputs, targets = sample_batch(train_data, length=sequence_length, batch_size=batch_size)
    inputs, targets = inputs.to(d()), targets.to(d())

    # Forward pass
    optimizer.zero_grad()
    output = model(inputs)
    output = output.view(-1, 256)
    targets = targets.view(-1)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    if (current_batch + 1) % print_interval == 0:
        print(f'Batch {current_batch + 1}: Training Loss = {loss.item()}')

    # Validation check
    if (current_batch + 1) % validation_interval == 0:
        val_loss, val_accuracy = validate(model, val_data, criterion, batch_size)
        print(f'Batch {current_batch + 1}: Validation Loss = {val_loss}, Accuracy = {val_accuracy}%')
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")
            counter = 0  # reset the counter
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break  # stop training if validation hasn't improved for 'patience' intervals

    pbar.update(1)

pbar.close()