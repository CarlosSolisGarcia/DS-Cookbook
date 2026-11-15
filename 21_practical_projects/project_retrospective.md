# Post-Project Retrospective: Turning Each Project into Lasting Skill

> **Folder:** 21_practical_projects · **Prev:** [final_solution.md](final_solution.md) · **Next:** back to [kaggle_workflow.md](kaggle_workflow.md) for the next project · **Related:** [iterative_improvement.md](../06_model_improvement/iterative_improvement.md), [README.md](../README.md) (how to extend the cookbook)

---

## 1. Purpose
Capture what you learned (techniques, mistakes, insights) while it is fresh. Compare your approach with the best solutions, and feed genuinely useful procedures back into this cookbook. The retrospective is what turns a sequence of projects into accumulated expertise.

## 2. When to use it
Within a week of finishing a competition or project, after the private leaderboard and winning solutions are published (for competitions).

## 3. Conceptual foundation
A good retrospective has four parts:
1. **Outcome vs expectations:** the final score/rank, CV vs private LB, what surprised you.
2. **Process review:** where time went, which stages were weak (recon, validation, features, ensembling), and which decisions were right or wrong *given the information you had at the time*. That is a decision-quality review, not a hindsight review.
3. **Gap analysis against the top solutions:** which ideas did the winners have that you did not? Was the gap one of knowledge, time or validation?
4. **Transfer:** reusable code, new techniques to practise, and cookbook updates (only procedures that are general and validated).

## 4. Assumptions and limitations
- Top solutions sometimes exploit leaks or massive compute. Separate transferable ideas from competition artefacts.
- One project is anecdotal evidence. Promote a technique to a cookbook recipe only when it has worked in more than one context, or is well supported in the literature.

## 5. Step-by-step procedure
1. **Collect the facts:** the experiment log, the CV–LB plot, the final private score, the time spent per stage.
2. **Answer the template questions** (§6).
3. **Read 3–5 top solution write-ups;** list their key ideas; classify each as known/unknown and used/unused.
4. **Pick at most three improvements** for your next project (skills to learn, process changes).
5. **Update the assets:** save reusable code to your utilities; add or amend cookbook recipes (a decision rule, a pitfall, a code pattern) with a reference to the project as evidence.
6. **Schedule practice** for one unknown technique (a small focused project).

## 6. Python implementation: retrospective helpers

```python
import numpy as np
import pandas as pd

# From the experiment log: where did the improvements come from?
log = pd.DataFrame({
    "id": ["e001", "e002", "e003", "e004", "e005", "e006"],
    "category": ["baseline", "model", "features", "features", "ensemble", "tuning"],
    "cv": [0.769, 0.788, 0.786, 0.801, 0.806, 0.807],
    "accepted": [True, True, False, True, True, True],
    "hours": [2, 1, 4, 6, 3, 5],
})
acc = log[log["accepted"]].copy()
acc["gain"] = acc["cv"].diff().fillna(0)
summary = acc.groupby("category").agg(gain=("gain", "sum")).join(log.groupby("category")["hours"].sum())
summary["gain_per_hour"] = summary["gain"] / summary["hours"]
print(summary.sort_values("gain_per_hour", ascending=False).round(4))

# CV vs private LB: how well did validation predict the final result?
subs = pd.DataFrame({"cv": [0.769, 0.788, 0.801, 0.806], "public": [0.79, 0.80, 0.80, 0.82], "private": [0.77, 0.79, 0.80, 0.80]})
print("corr(cv, private) =", round(subs["cv"].corr(subs["private"]), 3),
      "| corr(public, private) =", round(subs["public"].corr(subs["private"]), 3))
```

**Retrospective template:**
```text
RETROSPECTIVE — <project>, <date>
Result            : private rank/score ...; CV ... ; public ... (shake-up: ...)
Surprises         : ...
Time allocation   : recon x h, validation x h, features x h, models x h, ensembling x h, docs x h
What worked       : (with evidence from the log) ...
What did not      : ...
Decision quality  : decisions that were reasonable but unlucky / decisions that were poor given the info
Top-solution ideas: 1) ... [known? used?]  2) ...  3) ...
Gaps              : knowledge (e.g. GNN on molecules) / validation (e.g. group leakage missed) / time management
Next time         : max 3 concrete changes
Reusable assets   : code saved to ..., new cookbook entries: ...
Practice project  : technique ..., dataset ..., by <date>
```

**Cookbook update rule:** add a recipe change only if (a) it is generalisable beyond this dataset, (b) it was validated (in your experiments or the literature), and (c) it fits an existing recipe's structure (purpose, decision framework, pitfalls). Record the source:
```text
<!-- added 2026-11, evidence: <project name> retrospective; also: <reference> -->
```

## 7. Interpretation
- **Gain per hour by category** shows where your effort paid off. In tabular projects, features usually beat tuning. If tuning dominates your hours, rebalance next time.
- **corr(cv, private) vs corr(public, private):** if CV predicted the private LB better than the public LB did, your validation was sound. If not, validation design is your top learning goal.
- **Top-solution gap analysis** turns "they were better" into specific, learnable items.

## 8. Decision framework
- **IF** the private result was much worse than the CV → your validation did not mimic the test. Study how the winners validated; update [competition_validation.md](competition_validation.md) notes.
- **IF** the winners used a technique you did not know → plan a small practice project with that technique within a month.
- **IF** you knew the technique but lacked time → improve the time allocation (an earlier baseline, less tuning).
- **IF** a process step repeatedly causes trouble (e.g. late submission bugs) → automate it (verification scripts).
- **IF** an insight is dataset-specific → keep it in the project notes, not in the cookbook.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Skipping the retrospective | Repeating mistakes | Schedule it right after the project |
| Hindsight bias ("I should have known") | Wrong lessons | Judge decisions by the information available then |
| Copying winners' tricks without context | Non-transferable "lessons" | Classify ideas as general vs competition-specific |
| Too many resolutions | Nothing changes | Max 3 concrete changes |
| Bloating the cookbook | Loss of clarity | The update rule (general, validated, structured) |

## 10. Checklist
- [ ] Facts collected (log, CV–LB–private, time per stage).
- [ ] Template answered honestly.
- [ ] Top solutions read; ideas classified.
- [ ] ≤ 3 changes for next time.
- [ ] Reusable code saved; cookbook updated under the update rule.
- [ ] Practice project scheduled.

## 11. References
- Kahneman, D., Lovallo, D. & Sibony, O. (2011). "Before You Make That Big Decision…" *Harvard Business Review* 89(6). On decision-quality reviews.
- Ericsson, K. A., Krampe, R. T. & Tesch-Römer, C. (1993). "The role of deliberate practice in the acquisition of expert performance." *Psychological Review* 100(3), 363–406.
- Kaggle discussion forums: competition "solution write-up" threads (primary source for top-solution analysis).
