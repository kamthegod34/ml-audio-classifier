import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from data import dataloader_make
from model import small_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(logits, target):
    # logits is the raw inputs to final layer [B, C] - C is tensor with correct classes
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == target).float().mean().item() # turn bool into 0/1, then mean that and then into a python number
    return accuracy

def train_epoch(model, loader, optimizer, loss_func):
    model.train() # set to training mode
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    for inputs, targets in tqdm(loader, desc="Training"):
        # move to device
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # prevent accumulation of gradients
        logits = model(inputs) # forward pass
        # now calculate loss
        loss = loss_func(logits, targets)
        loss.backward() # backwards pass
        optimizer.step() # update weights
        batch_size = inputs.size(0) # might be smaller on final batch

        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, targets) * batch_size
        num_samples += batch_size

    total_acc /= num_samples
    total_loss /= num_samples

    return total_loss, total_acc  # epoch average loss and accuracy

@torch.no_grad()
def evaluate(model, loader, loss_func, description="validation"):
    model.eval() # set to evaluation mode
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    for inputs, targets in tqdm(loader, desc=description):
        # move to device
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs) # moves model forward
        loss = loss_func(logits, targets) # calculate loss
        batch_size = inputs.size(0) 

        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, targets) * batch_size
        num_samples += batch_size

    total_acc /= num_samples
    total_loss /= num_samples

    return total_loss, total_acc

def main():
    print(f"using device: {device}")

    # create data loaders each give you [inputs, targets]
    # wheere inputs is [B, 1, n_mels, time_frames] and targets is [B]
    train_loader, val_loader, _ = dataloader_make(bs=32, num_workers=2, pin=(device.type=="cuda"))
    model = small_CNN(n_classes=3).to(device) # create model then move to device
    
    # use cross entropy loss to calculate loss
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    # attempt to overfit on a tiny subset of data
    small_inputs, small_targets = next(iter(train_loader))
    small_inputs, small_targets = small_inputs.to(device), small_targets.to(device)

    for i in range(200): # 250 iterations should do the job
        optimizer.zero_grad()
        logits = model(small_inputs)
        loss = loss_func(logits, small_targets)
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            acc = accuracy(logits, small_targets)
            print(f"iteration {i+1}, loss: {loss.item():.3f}, accuracy: {acc:.3f}")
        

    best_val_acc = 0.0
    for epoch in range(1,16):
        training_loss, training_acc = train_epoch(model, train_loader, optimizer, loss_func)
        validation_loss, validation_acc = evaluate(model, val_loader, loss_func, description="validation")
        print(f"epoch number: {epoch:02d}, train: {training_loss:.3f}, {training_acc:.3f}, val: {validation_loss:.3f}, {validation_acc:.3f}")

        if validation_acc > best_val_acc:
            best_val_acc = validation_acc
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": validation_acc,
                "val_loss": validation_loss,
            }
            torch.save(checkpoint, "best_model.pt")
            print(f"new best model saved at epoch {epoch:02d} with validation accuracy {best_val_acc:.3f}")


if __name__ == "__main__":
    main()