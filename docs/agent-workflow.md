# How I Used Agents During the Project

I built this project as a hands-on learning exercise and used local coding agents to help me move faster while keeping the decisions and final review in my hands.

## How the work was organized

I split the research into a few focused jobs instead of asking one tool to do everything:

- one task checked the data timing and feature setup for leakage;
- separate tasks compared a few simple model choices;
- another task reviewed the saved predictions and recomputed the reported metrics;
- a final review brought the findings together and identified what was still uncertain.

I grouped these tasks into focused agent conversations around the same project workspace so their updates and saved results could be reviewed together. This made it easier to compare experiments on the same tickers, dates, features, and benchmark rather than treating every result as a separate answer.

## What I did myself

I chose the project question, set the scope, decided which results were strong enough to include, and checked the final code and charts. I also corrected a class-weight mismatch that showed up during review and kept the public repository limited to reproducible project files.

The agents helped with repetitive analysis and review, but they did not replace judgment. When an experiment looked slightly better than the benchmark, I checked its sample size and uncertainty before describing it as evidence. That is why the project reports small differences cautiously instead of presenting them as a reliable trading system.

## How I plan to use this workflow

As the project grows, I can use the same approach for separate next-day, next-week, and next-month direction experiments. Each task will have a written scope, a fixed evaluation plan, and a review step before anything is added to the public project. The private research workspace will hold exploratory notes, while the public repository will contain the clearest code, results, and explanations.

This workflow gave me practice with experiment design, reproducibility, code review, and communicating uncertainty—not just model training.
