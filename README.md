# Fine Tuning BERT on Trump Socials to Buy/Sell S&P 500

---

## **Project Description**

1.  Collects historical social media posts (tweets from 2017-2021) from the Trump Term 1
2.  Creates a labeled dataset by mapping each post to the forward performance of the S&P 500 (VOO).
3.  Fine-tunes DistilBERT to classify posts as `Bullish`, `Bearish`, or `Neutral`.
4.  Builds and runs a backtest of a simple trading strategy that toggles between VOO and cash based on the model's signals.
5.  Analyzes and compares the performance of the model against a "buy and hold" benchmark to determine if a verifiable alpha was present.


## Run The Notebook

You may see the notebook output in `test_model.ipynb`.

## Training Yourself

Use `uv` to install everything, and ensure your GPU size fits the batches.

This batch size fits on an RTX 3060 12GB.

Use torchrun to train, e.g. `torchrun --nproc_per_node=2 train.py`

Replace 2 with however many GPUs you wish to use.
