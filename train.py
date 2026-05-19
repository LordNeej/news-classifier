# train.py — Fine-tunes DistilBERT on AG News using a native PyTorch training loop.
# Uses AdamW + linear warmup, evaluates after every epoch, saves the best checkpoint.
# Targets 90%+ accuracy; typically reaches ~91-92% after 3-4 epochs.

import os
import math

# sklearn and pandas must be imported before torch so that PyArrow's DLL loads
# before CUDA DLLs are mapped into the process — reversing this order causes an
# access violation on Windows when pyarrow tries to load alongside CUDA.
from sklearn.metrics import accuracy_score, classification_report
from data import load_and_split_data, LABEL_NAMES   # also imports pandas early

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
MODEL_NAME    = "distilbert-base-uncased"
OUTPUT_DIR    = "./model"
NUM_LABELS    = 4
NUM_EPOCHS    = 4
BATCH_SIZE    = 16     # safe for 4 GB VRAM; effective batch = 16 × GRAD_ACCUM = 32
GRAD_ACCUM    = 2      # gradient accumulation steps
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.06
SEED          = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device) -> tuple[float, str]:
    """Run the model on `dataloader` and return (accuracy, classification_report)."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0)
    return acc, report


def main():
    set_seed(SEED)

    # ------------------------------------------------------------------ #
    # Device — use GPU if available, otherwise CPU
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data — num_workers=0 avoids Windows multiprocessing issues
    # ------------------------------------------------------------------ #
    print("\nPreparing datasets...")
    train_dataset, val_dataset, test_dataset = load_and_split_data(seed=SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # ------------------------------------------------------------------ #
    # Model — replace the pre-trained MLM head with a 4-class head
    # ------------------------------------------------------------------ #
    print(f"\nLoading {MODEL_NAME}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )
    model.to(device)

    # ------------------------------------------------------------------ #
    # Optimiser — AdamW with weight decay (bias & LayerNorm excluded)
    # ------------------------------------------------------------------ #
    no_decay = {"bias", "LayerNorm.weight"}
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_params, lr=LEARNING_RATE)

    # ------------------------------------------------------------------ #
    # LR scheduler — linear warmup then linear decay to 0
    # ------------------------------------------------------------------ #
    # Total optimizer steps = ceil(batches_per_epoch / grad_accum) × epochs
    steps_per_epoch  = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_steps      = steps_per_epoch * NUM_EPOCHS
    warmup_steps     = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Mixed-precision scalar — speeds training on GPU; no-op on CPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(
        f"\nTraining plan: {NUM_EPOCHS} epochs × {steps_per_epoch} opt-steps "
        f"({len(train_loader)} batches, grad_accum={GRAD_ACCUM})"
    )
    print(f"Warmup steps: {warmup_steps}  |  Total opt-steps: {total_steps}\n")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    best_val_acc  = 0.0
    best_epoch    = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss   = 0.0
        opt_step     = 0
        grad_step    = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Forward pass — use autocast for fp16 on GPU
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # Scale loss by accumulation steps so the effective gradient equals
                # the mean over the full accumulated batch
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            total_loss += outputs.loss.item()   # log unscaled loss for readability
            grad_step  += 1

            # Only update weights after GRAD_ACCUM micro-batches (or at end of epoch)
            if grad_step % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                # Gradient clipping prevents exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                # Print progress every 200 optimiser steps
                if opt_step % 200 == 0:
                    avg_loss = total_loss / grad_step
                    print(f"  Epoch {epoch} | opt-step {opt_step}/{steps_per_epoch} | loss {avg_loss:.4f}")

        # ---------------------------------------------------------------- #
        # Validation after each epoch
        # ---------------------------------------------------------------- #
        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_report = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}  train_loss={avg_train_loss:.4f}  val_acc={val_acc:.4f}")
        print(val_report)

        # Save checkpoint if this is the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            model.save_pretrained(OUTPUT_DIR)
            # Also save the tokenizer so predict.py can load everything from ./model
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  -> New best! Saved to {OUTPUT_DIR}  (val_acc={best_val_acc:.4f})\n")
        else:
            print(f"  -> No improvement (best so far: epoch {best_epoch}, val_acc={best_val_acc:.4f})\n")

    # ------------------------------------------------------------------ #
    # Final evaluation on the held-out test set
    # ------------------------------------------------------------------ #
    print("Loading best checkpoint for final test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR)
    model.to(device)
    test_acc, test_report = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"Test accuracy: {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"{'='*50}")
    print(test_report)
    print(f"\nBest model saved at: {OUTPUT_DIR}  (from epoch {best_epoch})")


if __name__ == "__main__":
    main()
