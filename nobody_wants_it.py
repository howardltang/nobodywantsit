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

def resolve_alias(name, aliases):
    """
    If name (case-insensitive) matches a key in the aliases dict,
    return the canonical target name. Otherwise return name unchanged.
    Aliases are stored as {alias_lower: canonical}.
    """
    return aliases.get(name.strip().lower(), name.strip())

def normalise_item(name, item_values):
    """Return existing canonical key for name if present, else name.strip()."""
    return canonical_item(name, item_values) or name.strip()

def normalise_player(name, player_profiles, aliases=None):
    """
    Return the canonical player name for `name`.
    1. Apply alias substitution (e.g. 'Neuv' → 'Neuvillette')
    2. Apply case-insensitive match against known profiles.
    3. Fall back to the stripped input.
    """
    resolved = resolve_alias(name, aliases) if aliases else name.strip()
    return canonical_player(resolved, player_profiles) or resolved

# ─────────────────────────────────────────────
# STATE  (persists across sessions)
# ─────────────────────────────────────────────

def load_state():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            state = json.load(f)
        # Ensure older state files have the aliases key
        state.setdefault("name_aliases", {})
        state.setdefault("decay_factor", DEFAULT_DECAY)
        return state
    return {
        "rounds":       [],
        "item_values":  {},
        "name_aliases": {},   # {alias_lowercase: canonical_name}
        "decay_factor": DEFAULT_DECAY,
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


DEFAULT_DECAY = 0.8   # weight of round N-1 relative to round N

class NWIModel:
    def __init__(self):
        self.item_history = {}   # item → {appearances: [(round_idx, count)], wins: int}
        self.n_rounds     = 0
        self.decay        = DEFAULT_DECAY

    def _rebuild_history(self, rounds):
        hist = defaultdict(lambda: {"appearances": [], "wins": 0})
        for r_idx, rd in enumerate(rounds):
            for item, players in rd["items"].items():
                count = len(players)
                hist[item]["appearances"].append((r_idx, count))
                if count == 1:
                    hist[item]["wins"] += 1
        self.item_history = dict(hist)
        self.n_rounds = len(rounds)

    def train(self, rounds, player_profiles, item_values, decay=None):
        """Build item history from all past rounds."""
        if decay is not None:
            self.decay = decay
        self._rebuild_history(rounds)
        n_appearances = sum(len(h["appearances"]) for h in self.item_history.values())
        n_wins        = sum(h["wins"]             for h in self.item_history.values())
        n_skipped     = sum(1 for h in self.item_history.values()
                            for _, c in h["appearances"] if c == 0)
        decay_str = f", decay={self.decay:.2f}" if self.decay < 1.0 else ", no decay"
        print(f"  [Model] Bayesian estimator loaded "
              f"({n_appearances} item-appearances, {n_wins} solo wins, "
              f"{n_skipped} zero-pick appearances, across {len(rounds)} rounds"
              f"{decay_str})")

    def _weighted_lam_others(self, appearances):
        """
        Compute a decay-weighted estimate of λ_others from a list of
        (round_idx, total_pick_count) tuples.

        Each appearance contributes:
          - others = max(count - 1, 0)  (skipped rounds → 0 others)
          - weight = decay ^ (n_rounds - 1 - round_idx)
            so the most recent round always has weight 1.0

        Returns (weighted_mean_others, effective_n) where effective_n is
        the sum of weights (used for prior blending).
        """
        if not appearances:
            return 0.0, 0.0

        total_w  = 0.0
        total_wo = 0.0
        for r_idx, count in appearances:
            age    = (self.n_rounds - 1) - r_idx   # 0 = most recent
            w      = self.decay ** age
            others = max(count - 1, 0)
            total_w  += w
            total_wo += w * others

        weighted_mean = total_wo / total_w if total_w > 0 else 0.0
        return weighted_mean, total_w

    def _p_solo_for_item(self, item, value, all_values_in_round, n_players, n_items):
        """
        P(I win | I pick this item) = e^(-λ_others).

        λ_others is a decay-weighted mean of (total_pickers - 1) per appearance,
        smoothed toward a value-based prior.
        """
        h = self.item_history.get(item)

        if h and h["appearances"]:
            obs_lam, eff_n = self._weighted_lam_others(h["appearances"])

            prior_lam = _value_lam_others(value, all_values_in_round, n_players)

            # Reduce prior influence when item has consistently zero picks
            all_skipped = all(c == 0 for _, c in h["appearances"])
            pseudo = 0.5 if all_skipped else PRIOR_STRENGTH
            w_obs  = eff_n / (eff_n + pseudo)

            lam_others = w_obs * obs_lam + (1 - w_obs) * prior_lam
            return _p_solo_from_lambda(lam_others)

        else:
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
            if h and h["appearances"]:
                n_app     = len(h["appearances"])
                n_picked  = sum(1 for _, c in h["appearances"] if c > 0)
                n_skipped = n_app - n_picked
                avg = (sum(c for _, c in h["appearances"] if c > 0) / n_picked
                       if n_picked else 0.0)
                skip_note = f", skipped {n_skipped}×" if n_skipped else ""
                hist_str  = f"{n_app} appearances, avg {avg:.1f} picks/round{skip_note}"
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

def browse_item_pickers(state):
    """
    Interactive loop: look up an item and see which players have picked it
    most often, with wins, collisions, and pick rate across appearances.
    """
    all_items = sorted(state["item_values"].keys())
    # Also include items seen in rounds but not in item_values (no price set)
    for rd in state["rounds"]:
        for item in rd["items"]:
            if item not in all_items:
                all_items.append(item)
    all_items = sorted(set(all_items))

    print(f"\n  {len(all_items)} known items. Type a name to look it up, or blank to return.")
    print(f"  (Partial names work — e.g. 'spice' will match 'spice melange')\n")

    while True:
        raw = input("  Item name > ").strip()
        if not raw:
            return

        # Exact match first (case-insensitive)
        canon = canonical_item(raw, state["item_values"])
        if canon is None:
            # Try partial match across all known items
            matches = [i for i in all_items if raw.lower() in i.lower()]
            if len(matches) == 1:
                print(f"  → Matched '{matches[0]}'")
                canon = matches[0]
            elif len(matches) > 1:
                print(f"  Multiple matches: {', '.join(matches[:8])}"
                      + ("..." if len(matches) > 8 else ""))
                print("  Be more specific, or type the full name.")
                continue
            else:
                print(f"  No item found matching '{raw}'.")
                sample = all_items[:10]
                print(f"  Known items: {', '.join(sample)}"
                      + ("..." if len(all_items) > 10 else ""))
                continue

        # Tally picks per player for this item
        pick_tally  = defaultdict(lambda: {"picks": 0, "wins": 0, "collisions": 0})
        total_appearances = 0
        total_picked_rounds = 0

        for rd in state["rounds"]:
            players = rd["items"].get(canon)
            if players is None:
                continue   # item wasn't in this round at all
            total_appearances += 1
            if len(players) == 0:
                continue   # item was on the board but nobody picked it
            total_picked_rounds += 1
            won = len(players) == 1
            for p in players:
                pick_tally[p]["picks"] += 1
                if won:
                    pick_tally[p]["wins"] += 1
                else:
                    pick_tally[p]["collisions"] += 1

        val = state["item_values"].get(canon, 0)
        skipped = total_appearances - total_picked_rounds
        print(f"\n  ── ITEM: {canon} {'─'*max(1, 48-len(canon))}")
        if val:
            print(f"  Price         : {val:,.0f}")
        print(f"  Appearances   : {total_appearances}")
        print(f"  Times picked  : {total_picked_rounds}"
              + (f"  (skipped {skipped}×)" if skipped else ""))

        if not pick_tally:
            print("  Nobody has ever picked this item.")
            print()
            continue

        # Sort by picks desc, then wins desc
        rows = sorted(pick_tally.items(), key=lambda x: (-x[1]["picks"], -x[1]["wins"]))
        print(f"\n  {'PLAYER':<24} {'PICKS':>6} {'WINS':>6} {'COLLISIONS':>11} {'PICK RATE':>10}")
        print(f"  {'─'*62}")
        for player, t in rows:
            pick_rate = t["picks"] / total_appearances if total_appearances else 0
            print(f"  {player[:23]:<24} {t['picks']:>6} {t['wins']:>6} "
                  f"{t['collisions']:>11} {pick_rate:>10.0%}")
        print()


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


def manage_aliases(state):
    """
    Interactive submenu for viewing, adding, and removing persistent name aliases.
    Aliases are stored as {alias_lowercase: canonical_name} and applied automatically
    whenever player names are read in during a round.
    """
    aliases = state.setdefault("name_aliases", {})

    while True:
        print("\n  ── NAME ALIASES ─────────────────────────────────────────")
        if aliases:
            for alias_low, canonical in sorted(aliases.items()):
                print(f"    '{alias_low}' → '{canonical}'")
        else:
            print("    (no aliases defined)")
        print()
        print("  [1] Add alias")
        print("  [2] Remove alias")
        print("  [0] Back")
        sub = input("\n  > ").strip()

        if sub == "0":
            return

        elif sub == "1":
            print("\n  Enter the alias (the name to auto-replace):")
            alias_raw = input("  Alias > ").strip()
            if not alias_raw:
                print("  (cancelled)")
                continue
            print(f"  Enter the canonical name '{alias_raw}' should map to:")
            canonical_raw = input("  Canonical > ").strip()
            if not canonical_raw:
                print("  (cancelled)")
                continue
            alias_key = alias_raw.lower()
            if alias_key == canonical_raw.lower():
                print("  Those are the same name — nothing to do.")
                continue
            aliases[alias_key] = canonical_raw
            save_state(state)
            print(f"  ✓ Alias added: '{alias_raw}' → '{canonical_raw}'")
            print(f"    Any future occurrence of '{alias_raw}' will be saved as '{canonical_raw}'.")

        elif sub == "2":
            if not aliases:
                print("  No aliases to remove.")
                continue
            print("\n  Enter the alias to remove:")
            raw = input("  Alias > ").strip()
            key = raw.lower()
            if key in aliases:
                canonical = aliases.pop(key)
                save_state(state)
                print(f"  ✓ Removed alias '{raw}' → '{canonical}'.")
            else:
                print(f"  No alias found for '{raw}'.")

        else:
            print("  Invalid choice.")


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
    participants = [normalise_player(p, player_profiles, state.get("name_aliases", {}))
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
        players = [normalise_player(p, player_profiles, state.get("name_aliases", {}))
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
    model.train(state["rounds"], player_profiles, state["item_values"],
                decay=state.get("decay_factor", DEFAULT_DECAY))

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
        model.train(state["rounds"], player_profiles, state["item_values"],
                    decay=state.get("decay_factor", DEFAULT_DECAY))
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
                print("  [4] Item pick history")
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

                elif sub == "4":
                    browse_item_pickers(state)

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
                my_player  = state.get("my_player")
                my_tag     = f" ({my_player})" if my_player else " (not set)"
                decay      = state.get("decay_factor", DEFAULT_DECAY)
                decay_tag  = f" ({decay:.2f})" + (" — recency OFF" if decay >= 1.0 else "")
                print("─" * 42)
                print(f"  [1] Set my player name{my_tag}")
                print("  [2] Merge player names")
                print("  [3] Manage name aliases")
                print(f"  [4] Recency decay factor{decay_tag}")
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

                elif sub == "3":
                    manage_aliases(state)

                elif sub == "4":
                    print(f"\n  Current decay factor: {decay:.2f}")
                    print("  How much weight to give older rounds relative to the most recent.")
                    print("  0.8 = round 5 sessions ago counts ~33% as much as the latest.")
                    print("  1.0 = all rounds weighted equally (no recency bias).")
                    print("  Enter a value between 0.5 and 1.0, or blank to keep:")
                    raw = input("  > ").strip()
                    if not raw:
                        print("  (unchanged)")
                    else:
                        try:
                            new_decay = float(raw)
                            if not (0.5 <= new_decay <= 1.0):
                                print("  [!] Value must be between 0.5 and 1.0.")
                            else:
                                state["decay_factor"] = new_decay
                                save_state(state)
                                model.train(state["rounds"], player_profiles,
                                            state["item_values"], decay=new_decay)
                                print(f"  ✓ Decay factor set to {new_decay:.2f}. Model retrained.")
                        except ValueError:
                            print("  [!] Please enter a number.")

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