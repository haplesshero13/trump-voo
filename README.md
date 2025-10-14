# Fine Tuning BERT on Trump Socials to Buy/Sell S&P 500

---

## Run The Notebook

You may see the notebook output in `test_model.ipynb`.

## Training Yourself

Use `uv` to install everything, and ensure your GPU size fits the batches.

This batch size fits on an RTX 3060 12GB.

Use torchrun to train, e.g. `torchrun --nproc_per_node=2 train.py`

Replace 2 with however many GPUs you wish to use.
