# Recovering Market-Moving Signals from Fine-Tuned BERT Using Transformer Interpretability

We fine tuned BERT on Trump Socials to Buy/Sell S&P 500. It did... pretty ok?

## Methods

1.  Collected historical social media posts (tweets from 2017-2021) from Term 1.
2.  Created a labeled dataset by mapping each post to the forward performance (closing price within 24h of post) of the S&P 500 (VOO).
3.  Fine-tuned DistilBERT to classify posts as `(Bullish, Bearish, Neutral)`.
4.  Built and ran a backtest of a simple trading strategy that toggles between VOO and cash based on the model's signals.
5.  Analyzed and compared the performance of the model against a "buy and hold" benchmark to determine if a verifiable alpha was present.
6.  Examined the various classifications that produced winning signals to understand what the model learned.

## Run The Notebook

You may see the notebook output in `test_model.ipynb`.

## Training Yourself

Use `uv` to install everything, and ensure your GPU size fits the batches.

This batch size fits on an RTX 3060 12GB.

Use torchrun to train, e.g. `torchrun --nproc_per_node=2 train.py`

Replace 2 with however many GPUs you wish to use.
