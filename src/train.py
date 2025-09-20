import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from .data import dataloader_make, CLASS_2_IDX
from .model import small_CNN
import yaml
import os
from pathlib import Path
from .utils import seedingSet as set_seed
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def accuracy(logits, target):
    # logits is the raw inputs to final layer [B, C] - C is tensor with correct classes
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == target).float().mean().item() # turn bool into 0/1, then mean that and then into a python number
    return accuracy

def train_epoch(model, loader, optimizer, loss_func, device, use_nb: bool):
    model.train() # set to training mode
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    for inputs, targets in tqdm(loader, desc="Training"):
        # move to device
        inputs, targets = inputs.to(device, non_blocking=use_nb), targets.to(device, non_blocking=use_nb)
        optimizer.zero_grad() # prevent accumulation of gradients
        logits = model(inputs) # forward pass
        # now calculate loss
        loss = loss_func(logits, targets)
        loss.backward() # backwards pass
        optimizer.step() # update weights
        batchSize = inputs.size(0) # might be smaller on final batch

        total_loss += loss.item() * batchSize
        total_acc += accuracy(logits, targets) * batchSize
        num_samples += batchSize

    total_acc /= num_samples
    total_loss /= num_samples

    return total_loss, total_acc  # epoch average loss and accuracy

@torch.no_grad()
def evaluate(model, loader, loss_func, device, use_nb: bool, description="validation"):
    model.eval() # set to evaluation mode
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    for inputs, targets in tqdm(loader, desc=description):
        # move to device
        inputs, targets = inputs.to(device, non_blocking=use_nb), targets.to(device, non_blocking=use_nb)
        logits = model(inputs) # moves model forward
        loss = loss_func(logits, targets) # calculate loss
        batchSize = inputs.size(0) 

        total_loss += loss.item() * batchSize
        total_acc += accuracy(logits, targets) * batchSize
        num_samples += batchSize

        # collect predictions and targets for confusion matrix later
        pred = logits.argmax(dim=1)
        all_preds.append(pred.cpu()) # move tensor to cpu before using numpy()
        all_targets.append(targets.cpu())

    pred_labels = torch.cat(all_preds).numpy() # sklearn likes numpy/lists but not tensors
    true_labels = torch.cat(all_targets).numpy()

    total_acc /= num_samples
    total_loss /= num_samples

    return total_loss, total_acc, true_labels, pred_labels

def write_confusion_matrix(true_labels, pred_labels, class_names, output_path="artifacts"):
    os.makedirs(output_path, exist_ok=True)

    # make classification text report
    report_text = classification_report(true_labels, pred_labels, target_names=class_names, 
                                    digits=4, zero_division=0) # zero division, prob unlikely but just in case
    
    with open(os.path.join(output_path, "val_report.txt"), "w", encoding="UTF-8") as f:
        f.write(report_text)

    # add optional label parameters in case a class doesnt appear we can still have correct matrix shape
    labels = list(range(len(class_names)))
    confucius_mtrx = confusion_matrix(true_labels, pred_labels, labels=labels) # pun on confucius

    fig, ax = plt.subplots(figsize=(5,4), dpi=150)
    im = ax.imshow(confucius_mtrx, interpolation="nearest")

    # give description to figure
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=45)

    # looping across matrix to add numbers in each cell, heatmap from imshow becomes easier to understand
    for i in range(confucius_mtrx.shape[0]):
        for j in range(confucius_mtrx.shape[1]):
            ax.text(j, i, int(confucius_mtrx[i,j]), ha="center", va="center", color="b") # using int is cleaner than float

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "val_confusion_matrix.png"), bbox_inches="tight") # bbox_inches removes whitespace
    plt.close(fig) # prevent memory leaks and just overall good practice

def main():
    seed = 69 # chosen by an honest dice roll :)
    set_seed(seed, deterministic=True) # helps reproducibility

    CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
    config = yaml.safe_load(CONFIG_PATH.read_text())


    # solve device
    dev = str(config["system"].get("device", "auto")).lower()
    if  dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev)
        
    # training parameters from config file
    batch_size = int(config["training"]["batch_size"])
    lr = float(config["training"]["lr"])
    epochs = int(config["training"]["epochs"])
    num_workers = int(config["system"]["num_workers"])
    pin_memory = bool(config["system"]["pin_memory"])
    drop_last = bool(config["training"].get("drop_last"))


    # absolute paths for csv files
    PROJECT_ROOT = Path(__file__).resolve().parents[1] # project root
    proc_dir = (PROJECT_ROOT / config["data"]["processed_dir"]).resolve()
    train_csv = proc_dir / config["data"]["train_csv"]
    val_csv   = proc_dir / config["data"]["val_csv"]
    test_csv  = proc_dir / config["data"]["test_csv"]

    # lil sanity check
    for p in [proc_dir, train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"neccessary file not found in path: {p}")


    print(f"using device: {device}")

    # check whether pin_memory should be used, even if its on, CLI only gives warning but its cleaner without it
    pin_memory = bool(config["system"]["pin_memory"]) and device.type == "cuda"

    # non_blocking speeds up transfer to gpu only if pinned memory is used
    use_nb = (device.type == "cuda") and pin_memory

    # create data loaders each give you [inputs, targets]
    # wheere inputs is [B, 1, n_mels, time_frames] and targets is [B]
    train_loader, val_loader, _ = dataloader_make(test_csv=str(test_csv), train_csv=str(train_csv), val_csv=str(val_csv),
    bs=batch_size, drop_last=drop_last, num_workers=num_workers, pin=pin_memory, seed=seed)

# ----------------------------------------------------------------------------------------------------------------------------
    # sanity check - use one batch and try to overfit it so we know model and training works
    sanity_model = small_CNN(n_classes=3).to(device)
    sanity_optimizer = Adam(sanity_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # attempt to overfit on a tiny subset of data
    small_inputs, small_targets = next(iter(train_loader))
    small_inputs, small_targets = small_inputs.to(device), small_targets.to(device)

    for i in range(200): # 250 iterations should do the job
        sanity_optimizer.zero_grad()
        logits = sanity_model(small_inputs)
        loss = loss_func(logits, small_targets)
        loss.backward()
        sanity_optimizer.step()
        if (i+1) % 50 == 0:
            acc = accuracy(logits, small_targets)
            print(f"iteration {i+1}, loss: {loss.item():.3f}, accuracy: {acc:.3f}")
            
    del sanity_model, sanity_optimizer, small_inputs, small_targets

# ----------------------------------------------------------------------------------------------------------------------------

    model = small_CNN(n_classes=3).to(device) # create model then move to device
    
    # use cross entropy loss to calculate loss
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
        

    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        training_loss, training_acc = train_epoch(model, train_loader, optimizer, loss_func, device, use_nb)
        # collect val loss/accuracy and also predictions/targets on validation set
        validation_loss, validation_acc, true_labels, pred_labels = evaluate(model, val_loader, loss_func, device, use_nb, description="validation")
        print(f"epoch number: {epoch:02d}, train: {training_loss:.3f}, {training_acc:.3f}, val: {validation_loss:.3f}, {validation_acc:.3f}")

        if validation_acc > best_val_acc:
            best_val_acc = validation_acc

            # best epoch artifacts, make special directory
            best_dir = os.path.join("artifacts", f"best_epoch_{epoch:03d}_acc_{validation_acc:.3f}_seed_{seed}")
            os.makedirs(best_dir, exist_ok=True)

            # save checkpoint both in epoch directory and then as a best_model.pt
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": validation_acc,
                "val_loss": validation_loss,
                "cfg": config,
                "seed": seed,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "class_to_idx": CLASS_2_IDX

            }
            torch.save(checkpoint, os.path.join(best_dir, "best_model.pt")) # save in epoch dir
            torch.save(checkpoint, "best_model.pt") # save in top level dir, always overwriting

            # ID order and dictionary order can be differnt so lets sort it out
            class_names = [k for k, v in sorted(CLASS_2_IDX.items(), key=lambda item: item[1])]

            # call confusion matrix
            write_confusion_matrix(true_labels, pred_labels, class_names=class_names, output_path=best_dir)

            # save some raw arrays, for extra analysis if needed
            np.save(os.path.join(best_dir, "true_labels.npy"), true_labels)
            np.save(os.path.join(best_dir, "pred_labels.npy"), pred_labels)
            print(f"new best model saved at epoch {epoch:02d} with validation accuracy {best_val_acc:.3f}")


if __name__ == "__main__":
    main()