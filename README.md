# Nobody Wants It — Pick Advisor

A command-line and web advisor for the game *Nobody Wants It*, where players secretly pick items from a shared list and only win an item if they picked it alone. The tool learns from every round you enter and estimates which item gives you the best chance of a solo win.

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
- `flask` (web UI only)

```
pip install numpy flask
```

---

## Usage

**Web UI (recommended)**

```
python web_nwi.py
```

Open `http://localhost:5001` in your browser.

**Command-line**

```
python nobody_wants_it.py
```

Both can be run from any directory — the script always finds `nwi_state.json` relative to its own location.

All data is saved automatically to `nwi_state.json` in the same directory. The file is created on first run.

---

## Web UI

The web UI runs on `http://localhost:5001` and provides the following pages via the sidebar:

### Play a Round

Steps through the round wizard:

1. **Enter participants and items** — participants as a comma-separated list; items one per line
2. **Set prices** — any item not seen before prompts for a price; typing an existing item name merges a typo with its canonical entry
3. **Recommendation** — ranked table of all items by EV; item prices are editable inline and rankings refresh automatically
4. **Enter results** — fill in who picked each item; leave blank for skipped items
5. **Winners** — solo winners are displayed; round is saved to history

A **Test mode** toggle in the round header runs through the full flow without saving any data.

### Player Leaderboard

Sortable table of all players showing picks, wins, win rate, and contrarian score. A filter bar narrows the list by name. Click **Details** on any row to expand an inline view of that player's round-by-round pick history.

### Item List

Sortable table of all known items. A filter bar narrows the list by name. Click **Details** on any row to expand an inline view of per-player pick stats for that item.

- **Item names** are editable inline (click to edit, Enter to save, Escape to cancel)
- **Prices** are editable inline with the same controls; prices display with thousand separators

### My Stats

Shows your personal round-by-round pick history — items picked, outcome, number of other pickers, and item value. Your player name must be set in Settings.

### Settings

| Setting | Description |
|---------|-------------|
| My Player Name | Records which player in the results is you. Enables personal annotations in the recommendation table. |
| Utility Mode | Scoring function used for recommendations. See Utility modes below. |
| Recency Decay Factor | Controls how much weight older rounds carry. `0.8` (default) = a round five sessions ago counts ~33% as much as the latest. `1.0` = all rounds weighted equally. |
| Merge Player Names | Rewrites all historical occurrences of one player name to another. Automatically records an alias so the old name maps to the new one in future rounds. |
| Merge Items | Combines two items into one, rewriting all historical round data. The source item is removed from the item list. |

---

## Menu structure (CLI)

```
[1] Play a round
[2] View statistics
      [1] Player leaderboard
      [2] Item prices
      [3] Browse player stats
      [4] Item pick history
      [0] Back
[3] My stats
[4] Settings
      [1] Set my player name
      [2] Merge player names
      [3] Manage name aliases
      [4] Recency decay factor
      [5] Utility mode
      [0] Back
[5] Quit
```

---

## Recommendation table

```
#    ITEM                             PRICE  P(SOLO)         EV  NOTES
1    spice melange                  290,000    89.2% ...        4 appearances, avg 1.0 picks/round, skipped 1×
2    mafia pinky ring                99,700    95.1% ...        4 appearances, avg 1.0 picks/round, skipped 3×  [YOU: 0W/1C in 1 picks]
...
★ = top pick by EV  |  EV = price × P(solo)  |  YOU: W=won, C=collision
```

The `[YOU: ...]` annotation appears on any item you have previously picked, showing your win/collision record on that specific item.

### Utility modes

The EV column can be scored three ways, configurable in Settings:

| Mode | Formula | Best for |
|------|---------|---------|
| `linear` (default) | `price × P(solo)` | Maximum raw expected value |
| `sqrt` | `√price × P(solo)` | Risk-adjusted — reduces dominance of extreme prices |
| `log` | `log(price) × P(solo)` | Kelly-style — best long-run growth over many rounds |

In backtesting, `sqrt` correctly identified actual solo winners ~6× more often than `linear`, because `linear` is dominated by ultra-high-value items that are always heavily contested.

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

The **Player Leaderboard** shows wins, win rate, and a contrarian score for all players. The contrarian score reflects how often a player avoids the items that others crowd onto. Click **Details** on any row to see that player's full round-by-round pick history.

**My Stats** shows the same view for your own player name, which can be set in Settings.

---

## State file

All data lives in `nwi_state.json`. It contains:

- `rounds` — list of rounds, each with an `items` dict mapping item names to lists of players who picked them (empty list = nobody picked it that round)
- `item_values` — dict of item name → price
- `my_player` — your player name, if set
- `name_aliases` — dict of `{alias_lowercase: canonical_name}` (managed automatically by player merges)
- `decay_factor` — recency decay weight (default `0.8`)
- `utility_mode` — scoring function for recommendations (`linear`, `sqrt`, or `log`)

The file is plain JSON and can be edited by hand if needed. It is excluded from version control (see `.gitignore`) as it contains personal player data.

---

## Files

| File | Description |
|------|-------------|
| `nobody_wants_it.py` | CLI script |
| `web_nwi.py` | Web UI — run this for browser access on port 5001 |
| `nwi_state.json` | Saved game data (created automatically, excluded from git) |
| `README.md` | This file |
