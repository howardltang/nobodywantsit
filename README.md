# Nobody Wants It — Pick Advisor

A command-line advisor for the game *Nobody Wants It*, where players secretly pick items from a shared list and only win an item if they picked it alone. The tool learns from every round you enter and estimates which item gives you the best chance of a solo win.

---

## How the game works

Each round, a list of items is revealed. Every player secretly picks one. If only one player picked a given item, they win it. If multiple players picked the same item, nobody wins it. The goal is to pick something valuable that nobody else wants.

---

## What this tool does

After each round you enter the participants, items, and results. The advisor builds up a history of how often each item gets picked, how often it gets ignored entirely, and who tends to pick what. Before each round it can recommend which item to go for, ranked by **Expected Value** — the item's price multiplied by your estimated probability of winning it solo.

### P(solo) — what it actually means

P(solo) is the probability that **if you pick this item, nobody else also picks it**. It is not a round-level prediction; it is a conditional probability from your own perspective as the decision-maker.

Technically: the number of *other* players who pick each item is modelled as a Poisson process with rate λ. P(solo) = e^(−λ), where λ is estimated from the item's full history including rounds where nobody picked it at all. An item skipped by everyone for four consecutive rounds will have a λ close to zero and a P(solo) close to 100%.

---

## Requirements

- Python 3.8+
- `numpy`

```
pip install numpy
```

---

## Usage

```
python nobody_wants_it.py
```

All data is saved automatically to `nwi_state.json` in the same directory. The file is created on first run.

---

## Menu structure

```
[1] Play a round
[2] View statistics
      [1] Player leaderboard
      [2] Item prices
      [3] Browse player stats
      [0] Back
[3] My stats
[4] Settings
      [1] Set my player name
      [2] Merge player names
      [0] Back
[5] Quit
```

---

## Playing a round

1. **Enter participants** — comma-separated list of player names for this round
2. **Enter items** — one per line, blank line to finish
3. **Set prices** — any item not seen before will prompt for a price; you can also type an existing item name to merge a typo with its canonical entry
4. **Get a recommendation** — optional ranked table of all items by EV, with your personal history on each item annotated
5. **Enter results** — format is `item: player1, player2, ...`; only enter items that were actually picked; items left out are recorded as skipped (which is itself useful data)

---

## Recommendation table

```
#    ITEM                             PRICE  P(SOLO)         EV  NOTES
1    spice melange                  290,000    89.2% ...        4 appearances, avg 1.0 picks/round, skipped 1×
2    mafia pinky ring                99,700    95.1% ...        4 appearances, avg 1.0 picks/round, skipped 3×  [YOU: 0W/1C in 1 picks]
...
★ = top pick by EV  |  EV = price × P(solo)  |  YOU: W=won, C=collision
```

The `[YOU: ...]` annotation appears on any item you have personally picked before, showing your win/collision record on that specific item.

---

## How the model learns

The estimator is a **Bayesian Poisson model** — no external ML libraries required beyond numpy.

For each item, it tracks:
- How many players picked it each round it appeared (excluding yourself, since you are the decision-maker)
- How many rounds it appeared but nobody picked it at all ("skipped" rounds)

The average number of other pickers per round — counting skipped rounds as zero — becomes λ. A small value-based prior is blended in for items with limited history, and that prior's influence diminishes quickly as data accumulates. After around four appearances, the observed history dominates almost entirely.

This means the model correctly assigns high P(solo) to consistently ignored items, even if they have moderate value.

---

## Player tracking

The **player leaderboard** (Statistics → Player leaderboard) shows win rate and a contrarian score for any player with three or more picks. The contrarian score reflects how often a player avoids the items that others crowd onto.

**Browse player stats** lets you look up any player by name (partial matching supported) and see their full pick history round by round — items picked, outcome, number of other pickers, and item value.

**My stats** shows the same view for your own player name, which can be set in Settings.

---

## Settings

**Set my player name** — records which player in the results is you. Once set, the recommendation table annotates items you have previously picked with your personal win/collision record.

**Merge player names** — if the same person appears under multiple names across different rounds (e.g. a nickname vs a full name), this option rewrites all historical occurrences of one name to another. Partial name matching is supported. A confirmation prompt shows how many occurrences will be affected before any changes are made.

---

## State file

All data lives in `nwi_state.json`. It contains:

- `rounds` — list of rounds, each with an `items` dict mapping item names to lists of players who picked them (empty list = nobody picked it that round)
- `item_values` — dict of item name → price
- `my_player` — your player name, if set

The file is plain JSON and can be edited by hand if needed.

---

## Files

| File | Description |
|------|-------------|
| `nobody_wants_it.py` | Main script — run this |
| `nwi_state.json` | Saved game data (created automatically) |
| `README.md` | This file |