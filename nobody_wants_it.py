"""
Nobody Wants It — ML Advisor
=============================
Fresh-start version. All data is learned from rounds you enter.

Round flow:
  1. Enter participants
  2. Enter item list
  3. Program asks price for any new items
  4. Optional: request a recommendation
  5. Enter results (item: player1, player2, ...)
  6. Model retrains and saves

Usage:
    python nobody_wants_it.py
"""

import json
import os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SAVE_FILE = "nwi_state.json"

# ─────────────────────────────────────────────
# CANONICAL NAME HELPERS
# Names are stored in their original casing but
# all lookups are case-insensitive via these helpers.
# ─────────────────────────────────────────────

def canonical_item(name, item_values):
    """Return the stored key whose lowercase matches name.lower(), or None."""
    low = name.strip().lower()
    for k in item_values:
        if k.lower() == low:
            return k
    return None

def canonical_player(name, player_profiles):
    """Return the stored key whose lowercase matches name.lower(), or None."""
    low = name.strip().lower()
    for k in player_profiles:
        if k.lower() == low:
            return k
    return None

def normalise_item(name, item_values):
    """Return existing canonical key for name if present, else name.strip()."""
    return canonical_item(name, item_values) or name.strip()

def normalise_player(name, player_profiles):
    """Return existing canonical key for name if present, else name.strip()."""
    return canonical_player(name, player_profiles) or name.strip()

# ─────────────────────────────────────────────
# STATE  (persists across sessions)
# ─────────────────────────────────────────────

def load_state():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            return json.load(f)
    return {
        "rounds": [],        # list of {items: {item: [players]}}
        "item_values": {},   # item -> price
    }

def save_state(state):
    with open(SAVE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ─────────────────────────────────────────────
# PLAYER PROFILES  (rebuilt each session)
# ─────────────────────────────────────────────

def build_player_profiles(rounds):
    """Recency-weighted behavioral profiles per player."""
    n = len(rounds)
    profiles = defaultdict(lambda: {"picks": 0, "wins": 0, "contrarian": 0.0})

    for r_idx, rd in enumerate(rounds):
        items = rd["items"]
        if not items:
            continue
        recency = (r_idx + 1) / n
        max_count = max(len(v) for v in items.values())

        for item, players in items.items():
            popularity = len(players) / max(max_count, 1)
            won = len(players) == 1
            for p in players:
                pr = profiles[p]
                pr["picks"] += 1
                if won:
                    pr["wins"] += 1
                if popularity < 0.5:
                    pr["contrarian"] += recency
                else:
                    pr["contrarian"] -= recency * 0.5

    for pr in profiles.values():
        t = pr["picks"]
        pr["win_rate"] = pr["wins"] / t if t > 0 else 0.0
        if t > 0:
            pr["contrarian"] /= t

    return dict(profiles)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def item_hist_stats(item, item_history):
    h = item_history.get(item)
    if not h or h["rounds"] == 0:
        return 1.5, 0.3, 0
    avg_picks = np.mean(h["pick_counts"]) if h["pick_counts"] else 1.5
    win_rate  = h["wins"] / h["rounds"]
    return avg_picks, win_rate, h["rounds"]

def make_features(item, value, all_values, item_history, all_players, player_profiles):
    sorted_vals = sorted(all_values)
    n_items = len(sorted_vals)
    percentile = sorted_vals.index(value) / max(n_items - 1, 1) if value in sorted_vals else 0.5
    mean_v = np.mean(all_values)
    std_v  = np.std(all_values) if len(all_values) > 1 else 1.0
    trap   = (value - mean_v) / (std_v + 1e-6)

    avg_picks, hist_win_rate, hist_rounds = item_hist_stats(item, item_history)

    win_rates   = [player_profiles.get(p, {}).get("win_rate", 0.1)   for p in all_players]
    contrarians = [player_profiles.get(p, {}).get("contrarian", 0.0) for p in all_players]
    activities  = [player_profiles.get(p, {}).get("picks", 1)        for p in all_players]
    n_players   = len(all_players)
    n_exp       = sum(1 for p in all_players if player_profiles.get(p, {}).get("picks", 0) >= 5)

    return [
        value,
        percentile,
        trap,
        avg_picks,       # historical avg pickers — a legitimate predictor
        hist_win_rate,   # historical solo-win rate for this item
        hist_rounds,     # how many times this item has appeared
        n_items,
        n_players,
        np.mean(win_rates)   if win_rates   else 0.1,
        np.std(win_rates)    if win_rates   else 0.0,
        np.mean(contrarians) if contrarians else 0.0,
        np.mean(activities)  if activities  else 1.0,
        n_exp,
    ]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class NWIModel:
    def __init__(self):
        self.clf     = None
        self.scaler  = StandardScaler()
        self.trained = False
        self.item_history = {}

    def _rebuild_history(self, rounds):
        hist = defaultdict(lambda: {"pick_counts": [], "wins": 0, "rounds": 0})
        for rd in rounds:
            for item, players in rd["items"].items():
                hist[item]["pick_counts"].append(len(players))
                hist[item]["rounds"] += 1
                if len(players) == 1:
                    hist[item]["wins"] += 1
        self.item_history = dict(hist)

    def train(self, rounds, player_profiles, item_values):
        self._rebuild_history(rounds)
        X, y = [], []
        rolling_hist = defaultdict(lambda: {"pick_counts": [], "wins": 0, "rounds": 0})

        for rd in rounds:
            items = rd["items"]
            all_vals = [item_values.get(i, 0) for i in items]
            all_players = list({p for players in items.values() for p in players})

            for item, players in items.items():
                feats = make_features(
                    item, item_values.get(item, 0), all_vals,
                    rolling_hist, all_players, player_profiles,
                )
                X.append(feats)
                y.append(1 if len(players) == 1 else 0)

            for item, players in items.items():
                rolling_hist[item]["pick_counts"].append(len(players))
                rolling_hist[item]["rounds"] += 1
                if len(players) == 1:
                    rolling_hist[item]["wins"] += 1

        if len(X) < 5:
            print("  [Model] Not enough data yet — using value-rank fallback.")
            self.trained = False
            return

        X = np.array(X, dtype=float)
        y = np.array(y)
        X_s = self.scaler.fit_transform(X)

        # Logistic Regression is more stable with small datasets.
        # GradientBoosting only kicks in once we have enough data to generalise.
        if len(X) >= 200 and y.sum() >= 30:
            self.clf = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=2,
                subsample=0.8, random_state=42
            )
            model_name = "GradientBoosting"
        else:
            # C controls regularisation strength — lower = more regularised.
            # With few samples we want strong regularisation to avoid overfitting.
            C = min(1.0, len(X) / 50.0)
            self.clf = LogisticRegression(max_iter=1000, C=C)
            model_name = f"LogisticRegression(C={C:.2f})"
        self.clf.fit(X_s, y)
        self.trained = True
        print(f"  [Model] {model_name} trained on {len(X)} picks "
              f"({int(y.sum())} wins / {int((~y.astype(bool)).sum())} collisions)")

    def score_items(self, items_with_values, all_players, player_profiles):
        all_vals = list(items_with_values.values())
        results  = []

        for item, value in items_with_values.items():
            avg_picks, hist_win_rate, hist_rounds = item_hist_stats(item, self.item_history)

            if not self.trained:
                p_solo = 0.5
            else:
                feats = make_features(
                    item, value, all_vals,
                    self.item_history, all_players, player_profiles,
                )
                fv   = np.array([feats], dtype=float)
                fv_s = self.scaler.transform(fv)
                p_solo = self.clf.predict_proba(fv_s)[0][1]

            ev = value * p_solo
            hist_str = (f"{hist_rounds} appearances, avg {avg_picks:.1f} picks/round"
                        if hist_rounds > 0 else "new item")
            results.append({"item": item, "value": value,
                             "p_solo": p_solo, "ev": ev, "history": hist_str})

        return sorted(results, key=lambda x: -x["ev"])

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════╗
║       NOBODY WANTS IT — ML Advisor v2.0          ║
║  Learns from every round you enter. Fresh start. ║
╚══════════════════════════════════════════════════╝""")

def show_recommendations(results):
    W = 66
    print(f"\n{'─'*W}")
    print(f"{'#':<4} {'ITEM':<28} {'PRICE':>9} {'P(SOLO)':>8} {'EV':>10}  NOTES")
    print(f"{'─'*W}")
    for i, r in enumerate(results):
        star = " ★" if i == 0 else ""
        print(f"{i+1:<4} {r['item'][:27]:<28} {r['value']:>9,.0f} "
              f"{r['p_solo']:>8.1%} {r['ev']:>10,.0f}  {r['history']}{star}")
    print(f"{'─'*W}")
    print("  ★ = top pick by Expected Value  |  EV = price × P(you win solo)\n")

def show_player_stats(player_profiles):
    rows = [(n, p) for n, p in player_profiles.items() if p.get("picks", 0) >= 3]
    rows.sort(key=lambda x: -x[1]["win_rate"])
    print(f"\n{'─'*52}")
    print(f"{'PLAYER':<22} {'PICKS':>6} {'WINS':>6} {'WIN%':>7} {'CNTRN':>7}")
    print(f"{'─'*52}")
    for name, p in rows[:30]:
        print(f"{name[:21]:<22} {p['picks']:>6} {p['wins']:>6} "
              f"{p['win_rate']:>7.1%} {p['contrarian']:>7.2f}")
    if not rows:
        print("  (no players with 3+ picks yet)")

# ─────────────────────────────────────────────
# ITEM NAME RESOLUTION  (typo / merge helper)
# ─────────────────────────────────────────────

def resolve_item_name(typed_name, known_items):
    """
    Check whether `typed_name` exactly matches a known item (case-insensitive).
    Returns the canonical name if found, otherwise None.
    `known_items` is any iterable of known item name strings.
    """
    lower = typed_name.lower()
    for k in known_items:
        if k.lower() == lower:
            return k
    return None

def prompt_merge_or_price(new_item, item_values):
    """
    Called when `new_item` is not in item_values.
    Asks the user for a price OR an existing item name to merge with.
    Returns (canonical_name, updated_item_values, was_merged).
    """
    known = list(item_values.keys())
    print(f"  price of {new_item}: ", end="", flush=True)
    while True:
        raw = input().strip()
        if not raw:
            print(f"    [!] Please enter a price or an existing item name to merge.")
            print(f"  price of {new_item}: ", end="", flush=True)
            continue

        # Check if the input matches an existing item name (merge intent)
        canonical = resolve_item_name(raw, known)  # resolve_item_name already lowercases
        if canonical is not None:
            print(f"    → Merged '{new_item}' into '{canonical}' (price: {item_values[canonical]:,.0f})")
            return canonical, item_values, True

        # Otherwise try to parse as a number
        try:
            price = float(raw.replace(',', ''))
            item_values[new_item] = price
            return new_item, item_values, False
        except ValueError:
            # Not a number and not a known item — give a hint
            print(f"    [!] '{raw}' is not a number and doesn't match any known item.")
            if known:
                # Show up to 5 closest-looking items as a hint
                suggestions = [k for k in known if raw.lower()[:4] in k.lower()][:5]
                if suggestions:
                    print(f"    Did you mean one of: {', '.join(suggestions)}?")
                else:
                    print(f"    Known items: {', '.join(known[:8])}{'...' if len(known) > 8 else ''}")
            print(f"  price of {new_item}: ", end="", flush=True)

def prompt_resolve_result_item(typed_item, items_this_round, item_values):
    """
    Called when an item entered during results is not in items_this_round.
    Lets the user re-type or link to an item from this round / the database.
    Returns the canonical item name to use (or None to skip).
    """
    all_known = list(set(items_this_round) | set(item_values.keys()))
    print(f"    [!] '{typed_item}' wasn't in this round's item list.")
    print(f"    Re-type the correct item name, or press Enter to skip this entry:")
    while True:
        raw = input("    > ").strip()
        if not raw:
            return None
        canonical = resolve_item_name(raw, all_known)
        if canonical is not None:
            print(f"    → Linked to '{canonical}'.")
            return canonical
        # Partial hint
        suggestions = [k for k in all_known if raw.lower()[:4] in k.lower()][:5]
        if suggestions:
            print(f"    Possible matches: {', '.join(suggestions)}")
        else:
            print(f"    This round's items: {', '.join(items_this_round)}")
        print(f"    Re-type the correct item name, or press Enter to skip:")

# ─────────────────────────────────────────────
# ROUND FLOW
# ─────────────────────────────────────────────

def run_round(state, model, player_profiles):
    # ── 1. Participants ──────────────────────────────────────
    print("\n┌─ ROUND SETUP ───────────────────────────────────────┐")
    print("│ Participants (comma-separated):                      │")
    print("└─────────────────────────────────────────────────────┘")
    raw = input("> ").strip()
    participants = [normalise_player(p, player_profiles)
                   for p in raw.split(',') if p.strip()]
    if not participants:
        print("  No participants entered. Cancelling.")
        return state, player_profiles
    print(f"  ✓ {len(participants)} participants.")

    # ── 2. Items ─────────────────────────────────────────────
    print("\nItems available this round (one per line, blank to finish):")
    items_this_round = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        items_this_round.append(normalise_item(line, state["item_values"]))
    if not items_this_round:
        print("  No items entered. Cancelling.")
        return state, player_profiles
    print(f"  ✓ {len(items_this_round)} items.")

    # ── 3. Prices for new items ───────────────────────────────
    item_values = state["item_values"]
    new_items   = [i for i in items_this_round if canonical_item(i, item_values) is None]
    # name_map: original entered name -> canonical name (after any merges)
    name_map = {i: i for i in items_this_round}

    if new_items:
        print(f"\n{len(new_items)} new item(s) — enter a price, or type an existing item name to merge:")
        for item in list(new_items):
            if canonical_item(item, item_values) is None:  # still not known (case-insensitive)
                canonical, item_values, was_merged = prompt_merge_or_price(item, item_values)
                if was_merged:
                    name_map[item] = canonical   # redirect this item to its canonical name
        state["item_values"] = item_values
        save_state(state)

    # Apply name_map so items_this_round uses canonical names going forward
    items_this_round = [name_map.get(i, i) for i in items_this_round]
    round_items = {i: item_values[i] for i in items_this_round}

    # ── 4. Recommendation ────────────────────────────────────
    print("\nWould you like a recommendation? (y/n)")
    if input("> ").strip().lower() == 'y':
        results = model.score_items(round_items, participants, player_profiles)
        show_recommendations(results)
        dangerous = [p for p in participants
                     if player_profiles.get(p, {}).get("win_rate", 0) >= 0.3
                     and player_profiles.get(p, {}).get("picks", 0) >= 5]
        if dangerous:
            print(f"  ⚠  High win-rate players this round: {', '.join(dangerous)}")

    # ── 5. Results ───────────────────────────────────────────
    print("\n┌─ ROUND RESULTS ─────────────────────────────────────┐")
    print("│ Format:  item: player1, player2, ...                 │")
    print("│ Enter only items that were picked. Blank to finish.  │")
    print("└─────────────────────────────────────────────────────┘")

    new_round = {"items": {}}
    while True:
        line = input("> ").strip()
        if not line:
            break
        if ':' not in line:
            print("  [!] Use format   item: player1, player2, ...")
            continue
        item_part, players_part = line.split(':', 1)
        item    = item_part.strip()
        players = [normalise_player(p, player_profiles)
                   for p in players_part.split(',') if p.strip()]
        if not item or not players:
            print("  [!] Need an item name and at least one player.")
            continue
        # Case-insensitive match against this round's item list
        match = next((i for i in items_this_round if i.lower() == item.lower()), None)
        if match is None:
            # Unknown item — prompt user to link or retype
            match = prompt_resolve_result_item(item, items_this_round, item_values)
            if match is None:
                print("  (entry skipped)")
                continue
        new_round["items"][match] = players

    if not new_round["items"]:
        print("  No results entered. Round not saved.")
        return state, player_profiles

    # Show winners
    print()
    any_win = False
    for item, players in new_round["items"].items():
        if len(players) == 1:
            print(f"  🏆  {players[0]} won {item}")
            any_win = True
    if not any_win:
        print("  (no solo winners this round)")

    # Save + retrain
    state["rounds"].append(new_round)
    save_state(state)

    player_profiles = build_player_profiles(state["rounds"])
    model.train(state["rounds"], player_profiles, state["item_values"])
    print(f"\n  ✓ Saved. Database now has {len(state['rounds'])} round(s).")

    return state, player_profiles

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print_banner()

    state = load_state()
    print(f"\n  Database: {len(state['rounds'])} round(s), "
          f"{len(state['item_values'])} known item price(s).")

    player_profiles = build_player_profiles(state["rounds"])
    model = NWIModel()
    if state["rounds"]:
        print("[Training on saved data...]")
        model.train(state["rounds"], player_profiles, state["item_values"])
    else:
        print("  No rounds yet — model will train after your first round.")

    print("\n  Ready!\n")

    while True:
        print("─" * 42)
        print("  [1] Play a round")
        print("  [2] View player statistics")
        print("  [3] View known item prices")
        print("  [4] Quit")
        choice = input("\n> ").strip()

        if choice == "1":
            state, player_profiles = run_round(state, model, player_profiles)

        elif choice == "2":
            show_player_stats(player_profiles)

        elif choice == "3":
            if not state["item_values"]:
                print("  No item prices stored yet.")
            else:
                print(f"\n  {'ITEM':<35} {'PRICE':>10}")
                print("  " + "─" * 47)
                for item, val in sorted(state["item_values"].items(), key=lambda x: -x[1]):
                    print(f"  {item[:34]:<35} {val:>10,.0f}")

        elif choice == "4":
            print("\n  Goodbye! 🎯")
            break

        else:
            print("  Invalid choice.")

if __name__ == "__main__":
    main()