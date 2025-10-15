import torch
import polars as pl
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

MODEL_NAME = 'answerdotai/ModernBERT-base'
MAX_LENGTH = 8192 # Increased to ModernBERT's 8k context window

# You may be able to increase this slightly depending on your GPU VRAM.
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4

OUTPUT_DIR = "./results-modernbert"
RUN_NAME = "modernbert-trump-tweet-voo"
FINAL_MODEL_PATH = "./modernbert-trump-tweet-voo"

data_path = "labeled_dataset.arrow"

print(f"Loading pre-processed data from {data_path}...")

try:
    labeled_dataset_pl = pl.read_ipc(data_path)
    labeled_dataset_pl = labeled_dataset_pl.with_columns(pl.col("label").cast(pl.Int64))
except FileNotFoundError:
    print(f"ERROR: Data file not found at '{data_path}'. Please run prepare.py first.")
    exit()

print("Tokenizing and splitting dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # Truncate to the model's max supported length.
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

full_dataset = Dataset.from_polars(labeled_dataset_pl)
tokenized_datasets = full_dataset.map(tokenize_function, batched=True)
dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42)

print(f"Training set size: {len(dataset['train'])}")
print(f"Evaluation set size: {len(dataset['test'])}")


print("Calculating class weights for unbalanced data...")
id2label = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
label2id = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}

train_labels = np.array(dataset['train']['label'])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Using Class Weights: BEARISH={weights_tensor[0]:.2f}, NEUTRAL={weights_tensor[1]:.2f}, BULLISH={weights_tensor[2]:.2f}")


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



print(f"Loading '{MODEL_NAME}' model with Flash Attention 2...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    attn_implementation="flash_attention_2", # Use Flash Attention 2
    dtype=torch.bfloat16                     # Recommended for performance
)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    # gradient_accumulation_steps=4,

    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,

    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,

    # Use bfloat16 for mixed-precision training, which works well with Flash Attention
    bf16=True,

    report_to="wandb",
)


trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

print("Starting model training...")
trainer.train()


if trainer.is_world_process_zero():
    print(f"Training complete. Saving best model to '{FINAL_MODEL_PATH}'...")
    trainer.save_model(FINAL_MODEL_PATH)
    print("✅ Done.")
