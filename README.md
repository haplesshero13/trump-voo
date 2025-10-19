# What Do Language Models Learn About Trump's Market-Moving Rhetoric?

## Introduction

Can we understand what language models learn about political communication and market movements? Previous research documented that Trump's tweets moved markets during his first term (2017-2021), but most studies focused on prediction accuracy rather than interpretation. His second term provides an interesting natural experiment: same communication style, different platform (Truth Social vs Twitter), different market conditions, and crucially - no COVID-19 lockdowns.

I fine-tuned two BERT-family models on Trump's first-term tweets labeled with VOO (S&P 500 ETF) price movements, then tested them on his second-term Truth Social posts from H1 2025. The goal wasn't to build a trading system, but to use interpretability tools to understand what linguistic features these models associate with market-moving communication.

The results raised more questions than they answered! I encourage you to dig into the data themselves on the following notebooks to see for yourself.

## Approach

### Data
- Training: Trump tweets from 2017-2021, labeled with next-day VOO returns (Bullish/Bearish/Neutral)
- Testing: Trump Truth Social posts from H1 2025 (completely out-of-sample, 4+ year gap, different platform)

### Base Models
- **DistilBERT**: Distilled version of BERT-base, 512 token context limit, trained with standard classification loss
- **ModernBERT**: 2024 architecture with 8k token context window, alternating local/global attention, optimized for Sharpe ratio on training data

### Encoder-only BERT architectures? In this economy??

Encoder models are designed for classification tasks and process entire inputs bidirectionally. They're faster and more efficient than decoder-only models (GPT-style) for prediction tasks like this, also allowing us to run many experiments quockly. ModernBERT specifically claims state-of-the-art performance on classification benchmarks, so we really wanted to see how well they'd do.

### Financial Backtesting
Used predicted labels to generate trading signals (Bullish = enter position, Bearish = exit, Neutral = hold), simulated with realistic 0.1% transaction fees using vectorbt.

## Results

Both models showed positive returns on the H1 2025 out-of-sample period:

- **DistilBERT**: 14.3% return (vs 3.6% buy-and-hold), 9 winning trades out of 14 closed
- **ModernBERT**: Similar positive performance, 18 winning trades (twice as many)

### Preliminary Findings

Both models were trained on Twitter data from 2017-2021 and tested on Truth Social posts from 2025 - a 4+ year gap with a platform change. The fact that either model showed positive results suggests there might be learnable patterns in how Trump's rhetoric relates to market sentiment, though we're cautious to jump to conclusions given the highly complex nature of financial markets and social media as a phenomenon in society at large.

ModernBERT's additional winning trades might come from capturing signals in the later portions of long posts that DistilBERT never even saw due to truncation. This suggests that signals in NLP aren't just about dealing with context length, but could also be impacted by input distribution changes over such sequence length differences.

**What We're Analyzing Next**

The detailed interpretability analysis is still in progress:
- Feature attribution: What specific words/phrases drive each model's predictions?
- Length-based comparison: Do models perform differently on short vs long posts?
- Overlap analysis: Where do they agree vs disagree, and why?
- Confidence patterns: Are predictions high-confidence or uncertain?

[Detailed word-level analysis with visualizations to come in follow-up. See the notebooks for preliminary results.]

### Limitations and Caveats
This project demonstrates several challenges in real-world interpretability research:

- **Single test period:** We've only evaluated on H1 2025. To validate that these patterns are robust rather than lucky, we need to test on H2 2025 and beyond.

- **BERT models are less popular to research:** Most NLP and large language model interpretability researchers have moved into analyzing very large language models that are fine tuned to have enormous general capabilities. BERT models, however, remain an extremely important workhorse and in particular ModernBERT is a timely upgrade to a relatively ancient architecture.

- **The n=1 problem:** Trump is unique. We can't validate these findings on other political figures or communication contexts, though ad we mention elsewhere, other indices like volatility are potentially very fruitful.

- **Potential overfitting and other learning challenges:** Even with out-of-sample testing, financial time series are noisy. The financial indicators in this project may just be coincidence.

- **Historical context is lost:** The historical differences between the first and second Trump terms are enormous, not the least of which includes Trump being banned on Twitter and subsequently launching Truth Social and posting there instead.

- **Financial NLP is adversarial:** Unlike toy benchmarks or synthetic datasets, financial applications face markets that arbitrage away predictable patterns, making validation difficult. As a real world problem, it may well be both difficult and too reductive to construct synthetic data, hence the reliance on backtesting.

**Future Work**

- Test on additional time periods
- Test on different sources (not just Twitter/Truth)
- Detailed feature attribution analysis comparing what each model learned
- Extend to volatility prediction (VIX) and other instruments, like bonds
- Compare to other models, for example powerful GPT LLMs
- Explore why different models agree and disagree: multiple valid patterns or spurious correlations?

**Code and Reproducibility**

Full code, data, and notebooks available in [the repo](https://github.com/haplesshero13/trump-voo)

[Fine-tuned models on HuggingFace](https://huggingface.co/yen-av)

---

**Conclusion**

This started as a weekend project to understand what BERT models learn about market-moving communication. The preliminary findings suggest that architectural choices such as context length in encoder-only models matter, and that different models can attend to different data and yet potentially learn similar things.

Whether these patterns represent genuine signal or lucky noise remains an open question, one that can only be answered by more sophisticated modeling testing. For instance, financial volatility is an extremely important index that is often easier to correlate with media content. But the interpretability analysis of *what* each model learned, regardless of whether the predictions generalize, reveals interesting insights about how language models develop representations of political rhetoric and economic sentiment.

More detailed analysis of specific linguistic features to come. For now, this case study demonstrates some of the challenges and opportunities in applying interpretability methods to real-world, domains like adversarial financial NLP and todays market-moving political rhetorical.

---

**Lets chat!**

This is preliminary work exploring interpretability in financial NLP.

Comments, suggestions, and critiques welcome.
