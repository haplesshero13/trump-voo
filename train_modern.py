import torch
import polars as pl
import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 8192

# Learning Rate:
# [1e-5, 2e-5, 3e-5]
LEARN_RATE = 3e-5

# Number of Epochs: [3, 5, 8]
EPOCHS = 5

# Effective Batch Size:
# [(4, 8), (8, 4), (16, 2)] (TRAIN_BATCH_SIZE, ACCUM_STEPS)
ACCUM_STEPS = 1
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE

OUTPUT_DIR = f"./results-modernbert-{TRAIN_BATCH_SIZE}"
RUN_NAME = f"modernbert-trump-tweet-voo-{TRAIN_BATCH_SIZE}"
FINAL_MODEL_PATH = f"./modernbert-trump-tweet-voo-{TRAIN_BATCH_SIZE}"

data_path = "labeled_dataset.arrow"

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

train_labels = np.array(dataset["train"]["label"])
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)

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
        loss = (
            loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            / ACCUM_STEPS
        )
        return (loss, outputs) if return_outputs else loss

print("Downloading S&P 500 data for backtesting...")
gspc_prices = None
# Ensure the 'date' column exists in your dataset
if 'date' in dataset['test'].column_names:
    eval_dates = sorted(dataset['test']['date'])
    start_date = pd.to_datetime(eval_dates[0]).strftime('%Y-%m-%d')
    # Add 1 day to the end date to ensure the last day's data is included
    end_date = (pd.to_datetime(eval_dates[-1]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        gspc_prices = yf.download('^GSPC', start=start_date, end=end_date)['Close']
        if gspc_prices.empty:
            print("Warning: No price data downloaded. Backtesting will be skipped.")
            gspc_prices = None
        else:
            print(f"Successfully downloaded {len(gspc_prices)} days of price data.")
    except Exception as e:
        print(f"Could not download price data: {e}. Backtesting will be skipped.")
else:
    print("Warning: 'date' column not found in dataset. Backtesting will be skipped.")


def compute_metrics(p):
    """
    Computes classification metrics and portfolio backtesting metrics.
    """
    preds = np.argmax(p.predictions, axis=1)

    # 1. Standard classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="weighted"
    )
    acc = accuracy_score(p.label_ids, preds)
    metrics = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # 2. VectorBT Backtesting
    if gspc_prices is not None:
        try:
            pred_dates = pd.to_datetime(dataset['test']['date'])

            # Map model outputs (0: BEAR, 1: NEUT, 2: BULL) to signals (-1: Short, 0: Hold, 1: Long)
            signals = np.select([preds == 0, preds == 2], [-1, 1], default=0)

            signals_sr = pd.Series(signals, index=pred_dates, name="signal").sort_index()
            # If multiple signals on the same day, take the last one
            signals_sr = signals_sr.groupby(signals_sr.index.date).last()
            signals_sr.index = pd.to_datetime(signals_sr.index)

            # Align signals with price data, filling missing days with 0 (Hold)
            aligned_signals = signals_sr.reindex(gspc_prices.index, fill_value=0)

            entries = aligned_signals == 1
            exits = aligned_signals == -1

            # Run the backtest
            pf = vbt.Portfolio.from_signals(gspc_prices, entries, exits, freq='D', init_cash=100_000)
            stats = pf.stats()

            # Add portfolio metrics to our results dictionary
            metrics['vbt_total_return'] = stats['Total Return [%]']
            metrics['vbt_sharpe_ratio'] = stats['Sharpe Ratio']
            metrics['vbt_max_drawdown'] = stats['Max Drawdown [%]']
            metrics['vbt_win_rate'] = stats['Win Rate [%]']

        except Exception as e:
            print(f"VectorBT evaluation failed: {e}")
            metrics['vbt_total_return'] = 0.0
    return metrics


print(f"Loading '{MODEL_NAME}' model with Flash Attention 2...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
)


training_args = TrainingArguments(
    optim="adamw_torch_fused", # improved optimizer
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=ACCUM_STEPS,
    learning_rate=LEARN_RATE,
    metric_for_best_model="eval_vbt_sharpe_ratio",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    bf16=True,
    report_to="wandb",
)


trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting model training...")
trainer.train()


if trainer.is_world_process_zero():
    print(f"Training complete. Saving best model to '{FINAL_MODEL_PATH}'...")
    trainer.save_model(FINAL_MODEL_PATH)
    print("✅ Done.")
