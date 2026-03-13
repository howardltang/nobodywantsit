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

# ─────────────────────────────────────────────
# BAYESIAN P(SOLO) ESTIMATOR
# ─────────────────────────────────────────────
#
# P(solo) = P(I win this item | I pick it)
#         = P(0 other players also pick it)
#
# We model the number of OTHER pickers as Poisson(λ_others), so:
#   P(solo | you pick) = e^(-λ_others)
#
# λ_others is estimated from history:
#   - Skipped rounds (nobody picked it) → 0 others that round
#   - Picked rounds → (total pickers - 1) others that round
# We average across all appearances, then Bayesian-smooth toward a
# value-based prior.
#
# This means items skipped every round → λ_others ≈ 0 → P(solo) ≈ 100%,
# which is exactly the right answer: "if I pick this, I'll be alone."

PRIOR_STRENGTH = 1.0   # pseudo-observations the prior represents (low = trust history fast)

def _p_solo_from_lambda(lam_others):
    """P(0 other pickers) = e^(-λ_others)."""
    return float(np.exp(-lam_others))

def _value_lam_others(value, all_values_in_round, n_players):
    """
    Prior λ_others for items with no history, based on value rank.
    Conservatively assumes 0–3 other pickers depending on value.
    This prior is quickly overridden once real data exists.
    """
    known = sorted(v for v in all_values_in_round if v > 0)
    if value > 0 and known:
        pct = known.index(value) / max(len(known) - 1, 1) if value in known else 0.5
    else:
        pct = 0.5
    # Range: 0.1 others (cheapest) to 3.0 others (most expensive)
    # Deliberately narrow so history dominates quickly.
    return 0.1 + pct * 2.9


class NWIModel:
    def __init__(self):
        self.item_history = {}   # item → {pick_counts, wins, rounds, skipped}

    def _rebuild_history(self, rounds):
        hist = defaultdict(lambda: {"pick_counts": [], "wins": 0, "rounds": 0, "skipped": 0})
        for rd in rounds:
            for item, players in rd["items"].items():
                hist[item]["rounds"] += 1
                if len(players) == 0:
                    hist[item]["skipped"] += 1
                else:
                    hist[item]["pick_counts"].append(len(players))
                    if len(players) == 1:
                        hist[item]["wins"] += 1
        self.item_history = dict(hist)

    def train(self, rounds, player_profiles, item_values):
        """Build item history from all past rounds (no ML fitting needed)."""
        self._rebuild_history(rounds)
        n_rounds_seen = sum(h["rounds"] for h in self.item_history.values())
        n_wins        = sum(h["wins"]   for h in self.item_history.values())
        n_skipped     = sum(h["skipped"] for h in self.item_history.values())
        print(f"  [Model] Bayesian estimator loaded "
              f"({n_rounds_seen} item-appearances, {n_wins} solo wins, "
              f"{n_skipped} zero-pick appearances, across {len(rounds)} rounds)")

    def _p_solo_for_item(self, item, value, all_values_in_round, n_players, n_items):
        """
        P(I win | I pick this item) = P(0 other players pick it) = e^(-λ_others).

        λ_others = avg number of OTHER players who pick this item per round,
        computed over all appearances (skipped rounds contribute 0 others).
        """
        h = self.item_history.get(item)

        if h and h["rounds"] > 0:
            n_obs    = h["rounds"]
            n_skipped = h["skipped"]

            # Others per round: picked rounds → (count-1), skipped rounds → 0
            others_list = [max(c - 1, 0) for c in h["pick_counts"]] + [0] * n_skipped
            obs_lam_others = np.mean(others_list)

            # Smooth toward value-based prior.
            # But if observed average is 0 (nobody has ever picked it),
            # reduce prior influence sharply — the data is clear.
            prior_lam = _value_lam_others(value, all_values_in_round, n_players)
            if obs_lam_others == 0:
                # All appearances were skipped: trust this signal strongly.
                # Give the prior weight of just 0.5 pseudo-observations.
                w_obs = n_obs / (n_obs + 0.5)
            else:
                w_obs = n_obs / (n_obs + PRIOR_STRENGTH)
            lam_others = w_obs * obs_lam_others + (1 - w_obs) * prior_lam

            return _p_solo_from_lambda(lam_others)

        else:
            # No history: use value rank as prior
            lam_others = _value_lam_others(value, all_values_in_round, n_players)
            return _p_solo_from_lambda(lam_others)

    def score_items(self, items_with_values, all_players, player_profiles):
        n_players = len(all_players)
        n_items   = len(items_with_values)
        all_vals  = list(items_with_values.values())
        results   = []

        for item, value in items_with_values.items():
            p_solo = self._p_solo_for_item(item, value, all_vals, n_players, n_items)
            ev = value * p_solo

            h = self.item_history.get(item)
            if h and h["rounds"] > 0:
                avg = np.mean(h["pick_counts"]) if h["pick_counts"] else 0.0
                skip_note = f", skipped {h['skipped']}×" if h["skipped"] > 0 else ""
                hist_str = (f"{h['rounds']} appearances, avg {avg:.1f} picks/round"
                            + skip_note)
            else:
                hist_str = "new item"

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

def show_recommendations(results, my_player=None, item_history=None):
    W = 72
    print(f"\n{'─'*W}")
    print(f"{'#':<4} {'ITEM':<28} {'PRICE':>9} {'P(SOLO)':>8} {'EV':>10}  NOTES")
    print(f"{'─'*W}")
    for i, r in enumerate(results):
        star = " ★" if i == 0 else ""
        notes = r['history']
        # Annotate with my personal history on this item
        if my_player and item_history:
            h = item_history.get(r['item'])
            if h:
                my_picks   = h.get("my_picks",   0)
                my_wins    = h.get("my_wins",    0)
                my_collide = h.get("my_collide", 0)
                if my_picks > 0:
                    tag = f"  [YOU: {my_wins}W/{my_collide}C in {my_picks} picks]"
                    notes += tag
        print(f"{i+1:<4} {r['item'][:27]:<28} {r['value']:>9,.0f} "
              f"{r['p_solo']:>8.1%} {r['ev']:>10,.0f}  {notes}{star}")
    print(f"{'─'*W}")
    print("  ★ = top pick by EV  |  EV = price × P(solo)  |  YOU: W=won, C=collision\n")

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

def build_my_item_history(rounds, my_player):
    """
    Returns a dict: item → {my_picks, my_wins, my_collide}
    tracking only rounds where my_player appeared in the results.
    """
    hist = defaultdict(lambda: {"my_picks": 0, "my_wins": 0, "my_collide": 0})
    my_lower = my_player.lower()
    for rd in rounds:
        for item, players in rd["items"].items():
            if any(p.lower() == my_lower for p in players):
                hist[item]["my_picks"] += 1
                if len(players) == 1:
                    hist[item]["my_wins"] += 1
                else:
                    hist[item]["my_collide"] += 1
    return dict(hist)

def show_my_stats(rounds, item_values, my_player):
    """Show a personal summary: rounds played, items picked, wins, collisions."""
    my_lower = my_player.lower()
    rounds_played, total_picks, total_wins, total_collide = 0, 0, 0, 0
    item_rows = []

    for rd in rounds:
        participated = False
        for item, players in rd["items"].items():
            if any(p.lower() == my_lower for p in players):
                participated = True
                total_picks += 1
                won = len(players) == 1
                total_wins    += int(won)
                total_collide += int(not won)
                item_rows.append((item, won, len(players), item_values.get(item, 0)))
        if participated:
            rounds_played += 1

    if total_picks == 0:
        print(f"\n  No recorded picks found for '{my_player}'.")
        return

    win_rate = total_wins / total_picks
    print(f"\n  ── MY STATS: {my_player} ──────────────────────────────────────")
    print(f"  Rounds played : {rounds_played}")
    print(f"  Items picked  : {total_picks}")
    print(f"  Solo wins     : {total_wins}  ({win_rate:.0%})")
    print(f"  Collisions    : {total_collide}")

    print(f"\n  {'ITEM':<38} {'OUTCOME':<10} {'OTHERS':>6} {'VALUE':>12}")
    print(f"  {'─'*70}")
    for item, won, n_pickers, val in item_rows:
        outcome = "🏆 WON" if won else "💥 collision"
        others  = n_pickers - 1
        print(f"  {item[:37]:<38} {outcome:<10} {others:>6} {val:>12,.0f}")
    print()

def show_player_detail(rounds, item_values, player_name):
    """Show full pick history for any named player, round by round."""
    p_lower = player_name.lower()
    rounds_played, total_picks, total_wins, total_collide = 0, 0, 0, 0
    item_rows = []

    for i, rd in enumerate(rounds):
        participated = False
        for item, players in rd["items"].items():
            if any(p.lower() == p_lower for p in players):
                participated = True
                total_picks += 1
                won = len(players) == 1
                total_wins    += int(won)
                total_collide += int(not won)
                item_rows.append((i + 1, item, won, len(players), item_values.get(item, 0)))
        if participated:
            rounds_played += 1

    if total_picks == 0:
        print(f"\n  No recorded picks found for '{player_name}'.")
        return

    win_rate = total_wins / total_picks
    print(f"\n  ── STATS: {player_name} {'─'*max(1, 50-len(player_name))}")
    print(f"  Rounds played : {rounds_played}")
    print(f"  Items picked  : {total_picks}")
    print(f"  Solo wins     : {total_wins}  ({win_rate:.0%})")
    print(f"  Collisions    : {total_collide}")

    print(f"\n  {'RD':>3}  {'ITEM':<38} {'OUTCOME':<14} {'OTHERS':>6} {'VALUE':>12}")
    print(f"  {'─'*77}")
    for rd_num, item, won, n_pickers, val in item_rows:
        outcome = "🏆 WON" if won else "💥 collision"
        others  = n_pickers - 1
        print(f"  {rd_num:>3}  {item[:37]:<38} {outcome:<14} {others:>6} {val:>12,.0f}")
    print()

def browse_player_stats(state, player_profiles):
    """Interactive loop: look up any player by name."""
    # Build a sorted list of all known players for display/autocomplete
    all_players = sorted(player_profiles.keys())
    print(f"\n  {len(all_players)} known players. Type a name to look them up, or blank to return.")
    print(f"  (Partial names work — e.g. 'gra' will match 'Graelk')\n")

    while True:
        raw = input("  Player name > ").strip()
        if not raw:
            return

        # Exact match first (case-insensitive)
        canon = canonical_player(raw, player_profiles)
        if canon:
            show_player_detail(state["rounds"], state["item_values"], canon)
            continue

        # Partial match
        matches = [p for p in all_players if raw.lower() in p.lower()]
        if len(matches) == 1:
            print(f"  → Matched '{matches[0]}'")
            show_player_detail(state["rounds"], state["item_values"], matches[0])
        elif len(matches) > 1:
            print(f"  Multiple matches: {', '.join(matches)}")
            print(f"  Be more specific, or type the full name.")
        else:
            print(f"  No player found matching '{raw}'.")
            # Show a few names as a hint
            sample = all_players[:12]
            print(f"  Known players: {', '.join(sample)}"
                  + ("..." if len(all_players) > 12 else ""))

def merge_player_names(state, player_profiles):
    """
    Interactive: rename all occurrences of one player name into another,
    across every round in the state file. Also updates my_player if affected.
    """
    all_players = sorted(player_profiles.keys())
    print(f"\n  {len(all_players)} known players.")
    print("  This will replace every occurrence of one name with another.")
    print("  Both partial and exact names are accepted.\n")

    def resolve(prompt):
        """Ask for a player name with partial-match support. Returns canonical or None."""
        while True:
            raw = input(f"  {prompt}").strip()
            if not raw:
                return None
            canon = canonical_player(raw, player_profiles)
            if canon:
                return canon
            matches = [p for p in all_players if raw.lower() in p.lower()]
            if len(matches) == 1:
                print(f"    → Matched '{matches[0]}'")
                return matches[0]
            elif len(matches) > 1:
                print(f"    Multiple matches: {', '.join(matches)} — be more specific.")
            else:
                print(f"    No player found matching '{raw}'.")
                sample = all_players[:10]
                print(f"    Known: {', '.join(sample)}" + ("..." if len(all_players) > 10 else ""))

    # Get the name to absorb (the one being removed)
    old_name = resolve("Merge FROM (name to remove) > ")
    if old_name is None:
        print("  (cancelled)")
        return state, player_profiles

    # Get the canonical target
    print(f"  Merging '{old_name}' into …")
    new_name = resolve("Merge INTO (name to keep)  > ")
    if new_name is None:
        print("  (cancelled)")
        return state, player_profiles

    if old_name.lower() == new_name.lower():
        print("  Those are the same name — nothing to do.")
        return state, player_profiles

    # Confirm
    print(f"\n  ⚠  Rename '{old_name}' → '{new_name}' across all {len(state['rounds'])} rounds?")
    print("  Type 'yes' to confirm, or anything else to cancel:")
    if input("  > ").strip().lower() != "yes":
        print("  (cancelled)")
        return state, player_profiles

    # Apply across all rounds
    old_lower = old_name.lower()
    changed = 0
    for rd in state["rounds"]:
        for item, players in rd["items"].items():
            for i, p in enumerate(players):
                if p.lower() == old_lower:
                    players[i] = new_name
                    changed += 1

    # Update my_player if it was the old name
    if state.get("my_player", "").lower() == old_lower:
        state["my_player"] = new_name
        print(f"  ✓ Also updated 'my player' to '{new_name}'.")

    save_state(state)
    player_profiles = build_player_profiles(state["rounds"])
    print(f"  ✓ Done. {changed} occurrence(s) renamed '{old_name}' → '{new_name}'.")
    return state, player_profiles


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
        my_player = state.get("my_player")
        my_item_hist = build_my_item_history(state["rounds"], my_player) if my_player else None
        show_recommendations(results, my_player=my_player, item_history=my_item_hist)
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

    # Require at least one picked item to consider the round valid.
    # (Entering zero picks could be an accidental blank submit.)
    if not new_round["items"]:
        print("  No results entered. Round not saved.")
        return state, player_profiles

    # Fill in unpicked items with empty player lists so the model learns
    # that those items were available but nobody chose them.
    for item in items_this_round:
        if item not in new_round["items"]:
            new_round["items"][item] = []

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
        my_player = state.get("my_player")
        my_tag = f" ({my_player})" if my_player else " (not set)"
        print("─" * 42)
        print("  [1] Play a round")
        print("  [2] View statistics")
        print(f"  [3] My stats{my_tag}")
        print("  [4] Settings")
        print("  [5] Quit")
        choice = input("\n> ").strip()

        if choice == "1":
            state, player_profiles = run_round(state, model, player_profiles)

        elif choice == "2":
            # ── Statistics submenu ────────────────────────────────
            while True:
                print("─" * 42)
                print("  [1] Player leaderboard")
                print("  [2] Item prices")
                print("  [3] Browse player stats")
                print("  [0] Back")
                sub = input("\n> ").strip()

                if sub == "1":
                    show_player_stats(player_profiles)

                elif sub == "2":
                    if not state["item_values"]:
                        print("  No item prices stored yet.")
                    else:
                        print(f"\n  {'ITEM':<35} {'PRICE':>10}")
                        print("  " + "─" * 47)
                        for item, val in sorted(state["item_values"].items(),
                                                key=lambda x: -x[1]):
                            print(f"  {item[:34]:<35} {val:>10,.0f}")

                elif sub == "3":
                    browse_player_stats(state, player_profiles)

                elif sub == "0":
                    break
                else:
                    print("  Invalid choice.")

        elif choice == "3":
            if not my_player:
                print(f"  No player name set. Go to Settings to set it.")
            else:
                show_my_stats(state["rounds"], state["item_values"], my_player)

        elif choice == "4":
            # ── Settings submenu ──────────────────────────────────
            while True:
                my_player = state.get("my_player")
                my_tag = f" ({my_player})" if my_player else " (not set)"
                print("─" * 42)
                print(f"  [1] Set my player name{my_tag}")
                print("  [2] Merge player names")
                print("  [0] Back")
                sub = input("\n> ").strip()

                if sub == "1":
                    print(f"\n  Current name: {my_player or '(none)'}")
                    print("  Enter your player name (as it appears in results), or blank to keep:")
                    raw = input("  > ").strip()
                    if raw:
                        canonical = canonical_player(raw, player_profiles)
                        name = canonical if canonical else raw
                        state["my_player"] = name
                        save_state(state)
                        print(f"  ✓ My player set to '{name}'.")
                    else:
                        print("  (unchanged)")

                elif sub == "2":
                    state, player_profiles = merge_player_names(state, player_profiles)

                elif sub == "0":
                    break
                else:
                    print("  Invalid choice.")

        elif choice == "5":
            print("\n  Goodbye! 🎯")
            break

        else:
            print("  Invalid choice.")

if __name__ == "__main__":
    main()