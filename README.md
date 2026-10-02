# Stock Market Performance Analysis

### Exploring stock performance, testing predictions, and learning what holds up

**Ryan Dominguez** · Computer Science, Montclair State University · Expected graduation: December 2027

I wanted to see whether simple patterns in stock prices could help predict the next trading day's direction. I started by comparing eight companies with SPY, then tested whether recent returns, volume, and volatility could beat a basic "always up" guess.

This is a Python project I’m building while studying Computer Science at Montclair State University. It combines historical performance analysis with a small, time-based machine-learning experiment.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Git

**At a glance:** 8 stocks + SPY · 24,894 historical observations · 15 features · 5 charts · 8 integrity tests

[Results](#what-i-found) · [Methods](#how-i-tested-it) · [My workflow](#how-i-organized-the-work) · [Run the project](#run-it-locally)

![Historical growth of $100](figures/historical_growth.png)

## What the project does

- Downloads and checks daily market data for eight stocks and SPY.
- Compares growth, volatility, maximum drawdown, and return correlations.
- Builds 15 features from recent prices, trading volume, and market movement.
- Tests Logistic Regression and Random Forest against a simple baseline.
- Saves charts, predictions, and evaluation results so the analysis can be reproduced.

The main dataset has **24,894 observations from 2015–2025** for AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, XOM, WMT, and SPY. It comes from [Yahoo Finance through yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), using adjusted OHLC prices. SPY is the benchmark; predictions cover the eight individual stocks.

## What I found

AMZN had the highest annualized growth in this group, about **27.90%**, but its largest peak-to-trough decline was **56.15%**. SPY grew about **13.43% annually** with lower volatility than any of the selected stocks. Comparing the returns alone would miss that difference in risk.

Predicting the next day was harder:

| Experiment | Dates evaluated | Model accuracy | Always-up accuracy |
|---|---|---:|---:|
| Original Logistic Regression | 2024–2025 | 48.55% | 54.69% |
| Accuracy-focused follow-up | January–September 2026 | 51.88% | 51.48% |

The first model performed worse than the basic guess. The follow-up improved on its same-period baseline by **0.40 percentage points**, but it predicted down only **10 times out of 1,488 predictions**. Its approximate 95% interval for the improvement was **0.00 to +0.94 points**, so I can’t call that a reliable advantage.

The baseline changes because these are different periods. Each model needs to be compared with always-up on the same dates.

[Original results](results/model_metrics.csv) · [Follow-up results](results/followup/holdout_metrics.csv) · [Risk and return chart](figures/risk_return.png)

## How I tested it

The model uses information available **after today's close** to classify whether the next session closes higher. It predicts direction, not an exact price.

The original experiment trained on 2015–2021, chose a model using 2022–2023, and tested it on 2024–2025. It focused on balanced accuracy, which gives up and down days equal weight. The follow-up used overall accuracy, four yearly validation windows, and three fixed decision thresholds before its first 2026 evaluation.

The data is never randomly shuffled into training and test sets. Rolling features look backward, scaling is fitted on training data, and rows are removed if their next-day label crosses a split boundary. Eight tests check data integrity, feature timing, labels, and parts of the uncertainty calculation.

Both evaluation periods have now been inspected. Further experiments on them are exploratory, not new independent confirmation.

[Detailed methods and metric definitions](docs/methods.md) · [Follow-up experiment](docs/followup.md)

## What I learned

The biggest lesson was that a model needs to beat a meaningful baseline. A score above 50% can sound good until you check how often stocks went up without any model involved.

I also learned that choosing an evaluation metric changes what "better" means. The original experiment gave both directions equal weight. The follow-up focused on the percentage of correct predictions. Keeping those goals separate made the comparison much clearer.

## How I organized the work

Alongside the analysis, I set up a local Hermes agent workspace on my computer with agent profiles, task instructions, and separate experiment folders. I organized the agents into a project group and used a task board to keep track of their assignments and findings. That gave the project a repeatable workflow as I started asking more questions.

I used Codex to help build and debug the Python code and manage the Hermes work with me. My role has been choosing the questions, setting the scope, working through the results, and deciding what to investigate next. The setup lets me hand off repeatable tasks while I work on understanding the analysis and shaping the project.

- **Experiment agents** compared specific ideas, such as shorter training histories or a fixed tree model, in separate folders.
- **Audit and review agents** checked data timing, baseline comparisons, and whether the conclusions matched the evidence.
- **Codex and I** used the findings to plan follow-ups and prepare the parts suitable for this portfolio.

The review step mattered. One experiment used balanced class weights when the intended comparison was unweighted. That mismatch was caught, the experiment was rerun, and the earlier result was excluded. It was a useful reminder that generated code and reports still need to be checked.

I plan to keep using this workflow as the project grows: give agents a narrow question, keep the settings and results together, review the evidence with Codex, and add only the finished work here. Internal coordination stays in a separate private repository. I'm continuing to learn the modeling details as I build, so understanding and explaining each addition is part of the work.

## Run it locally

Use Python 3.12. Internet access is needed for the first download; subsequent runs use a local cache. No notebook or API key is required.

```powershell
git clone https://github.com/ryandom-codes/stock-market-performance-analysis.git
cd stock-market-performance-analysis
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\market_analysis.py
.\.venv\Scripts\python.exe src\beat_the_baseline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

In VS Code, select `.venv\Scripts\python.exe` as the interpreter. The scripts save their outputs to `figures/` and `results/`. Downloaded prices stay in the ignored `data/` folder; package versions and source fingerprints accompany the results. Provider revisions can change a new download.

```text
src/       Python analysis and follow-up experiment
tests/     Eight checks for data and evaluation integrity
figures/   Performance, risk, correlation, and model charts
results/   Saved metrics, predictions, and run details
docs/      Methods and follow-up design
data/      Local downloads, excluded from Git
```

## Limitations and next steps

These are familiar companies selected retrospectively, not a complete historical stock universe. Adjusted data can be revised, stocks share market shocks, and classification accuracy doesn’t establish trading returns. The project doesn’t simulate transaction costs or actual order execution.

Next, I want to finish reviewing the additional experiments and investigate whether these signals are more useful for forecasting risk than direction. Any claim of dependable prediction would need a prospective test with predictions recorded before the outcomes are known.
