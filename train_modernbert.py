import os
import polars as pl
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import torch
import numpy as np

data_path = "labeled_dataset.arrow"

print(f"Loading pre-processed data from {data_path}...")

try:
    labeled_dataset_pl = pl.read_ipc(data_path)
    labeled_dataset_pl = labeled_dataset_pl.with_columns(pl.col("label").cast(pl.Int64))
except FileNotFoundError:
    print(f"ERROR: Data file not found at '{data_path}'. Please run prepare.py first.")
    exit()

print("Tokenizing and splitting dataset...")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


full_dataset = Dataset.from_polars(labeled_dataset_pl)
tokenized_datasets = full_dataset.map(tokenize_function, batched=True)
dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42)

print(f"Training set size: {len(dataset['train'])}")
print(f"Evaluation set size: {len(dataset['test'])}")


print("Calculating class weights for unbalanced data...")
id2label = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
label2id = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}

train_labels = np.array(dataset["train"]["label"])
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)

# Note: When using torchrun, the trainer handles moving tensors to the correct device
weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(
    f"Using Class Weights: BEARISH={weights_tensor[0]:.2f}, NEUTRAL={weights_tensor[1]:.2f}, BULLISH={weights_tensor[2]:.2f}"
)


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


print("Loading 'ModernBERT-base' model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=3, id2label=id2label, label2id=label2id
)


training_args = TrainingArguments(
    output_dir="./results",
    run_name="modernbert-trump-multi-gpu",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=250,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    report_to="wandb",
)


trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

print("Starting model training...")
trainer.train()


if trainer.is_world_process_zero():
    final_model_path = "./modernbert-trump-tweet-voo"
    print(f"Training complete. Saving best model to '{final_model_path}'...")
    trainer.save_model(final_model_path)
    print("✅ Done.")
