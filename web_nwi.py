"""
Nobody Wants It — Web UI
Run:   python web_nwi.py
Open:  http://localhost:5000
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict

from flask import Flask, request, jsonify, render_template

sys.path.insert(0, str(Path(__file__).parent))
import math
from nobody_wants_it import (
    load_state, save_state, build_player_profiles, NWIModel,
    canonical_item, canonical_player, normalise_item, normalise_player,
    build_my_item_history, DEFAULT_DECAY, DEFAULT_UTILITY, UTILITY_LABELS,
    parse_multiplier,
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------------------------------------------------------------------------
# Global state  (single-user local app)
# ---------------------------------------------------------------------------
_state          = load_state()
_player_profiles = build_player_profiles(_state["rounds"])
_model          = NWIModel()
if _state["rounds"]:
    _model.train(_state["rounds"], decay=_state.get("decay_factor", DEFAULT_DECAY))

# Pending round being constructed across steps
_pending: dict = {}   # {participants, items}


def _retrain():
    global _player_profiles
    _player_profiles = build_player_profiles(_state["rounds"])
    _model.train(_state["rounds"], decay=_state.get("decay_factor", DEFAULT_DECAY))


# ---------------------------------------------------------------------------
# API — Round
# ---------------------------------------------------------------------------

@app.route("/api/round/start", methods=["POST"])
def api_round_start():
    """Step 1: parse participants + items, identify new items needing prices."""
    global _pending
    data = request.get_json()

    raw_parts    = [p.strip() for p in data.get("participants", "").split(",") if p.strip()]
    aliases      = _state.get("name_aliases", {})
    participants = [normalise_player(p, _player_profiles, aliases) for p in raw_parts]
    raw_items = [l.strip() for l in data.get("items", "").splitlines() if l.strip()]

    # Parse multipliers before normalising names
    parsed    = [parse_multiplier(l) for l in raw_items]
    items     = [normalise_item(base, _state["item_values"]) for base, _ in parsed]
    mults_raw = {normalise_item(base, _state["item_values"]): mult
                 for base, mult in parsed if mult > 1}

    if not participants:
        return jsonify({"error": "Please enter at least one participant."}), 400
    if not items:
        return jsonify({"error": "Please enter at least one item."}), 400

    new_items = [i for i in items if canonical_item(i, _state["item_values"]) is None]
    _pending  = {"participants": participants, "items": items, "multipliers": mults_raw}
    return jsonify({"new_items": new_items, "participants": participants,
                    "items": items, "multipliers": mults_raw})


@app.route("/api/round/prices", methods=["POST"])
def api_round_prices():
    """Step 2: receive prices (or merge targets) for new items."""
    data   = request.get_json()
    prices = data.get("prices", {})

    mults = _pending.setdefault("multipliers", {})

    for item, raw_val in prices.items():
        raw_val = str(raw_val).strip()
        try:
            price = float(raw_val.replace(",", ""))
            _state["item_values"][item] = price
        except ValueError:
            canon = canonical_item(raw_val, _state["item_values"])
            if canon:
                # Replace occurrences in pending items; carry over multiplier
                if item in mults:
                    mults[canon] = mults.pop(item)
                _pending["items"] = [canon if i == item else i for i in _pending["items"]]
            else:
                return jsonify({"error": f"'{raw_val}' is not a number and doesn't match a known item."}), 400

    # Deduplicate after merges
    seen, deduped = set(), []
    for i in _pending["items"]:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    _pending["items"] = deduped

    save_state(_state)
    return jsonify({"ok": True, "items": _pending["items"]})


@app.route("/api/round/update_price", methods=["POST"])
def api_round_update_price():
    """Update an item's price mid-round (before results are saved)."""
    data  = request.get_json()
    item  = data.get("item", "").strip()
    price = data.get("price")
    if not item:
        return jsonify({"error": "No item specified."}), 400
    try:
        price = float(str(price).replace(",", ""))
        if price < 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid price."}), 400
    _state["item_values"][item] = price
    save_state(_state)
    return jsonify({"ok": True})


@app.route("/api/round/recommend", methods=["POST"])
def api_round_recommend():
    if not _pending.get("items"):
        return jsonify({"error": "No pending round."}), 400

    utility  = _state.get("utility_mode", DEFAULT_UTILITY)
    mults    = _pending.get("multipliers", {})
    round_items = {i: _state["item_values"].get(i, 0) * mults.get(i, 1)
                   for i in _pending["items"]}
    my_player  = _state.get("my_player")
    results    = _model.score_items(round_items, _pending["participants"],
                                    utility=utility, my_player=my_player)
    my_hist    = build_my_item_history(_state["rounds"], my_player) if my_player else {}

    mb_items = _mystery_box_items()
    ranked = []
    for r in results:
        row = {
            "rank":       len(ranked) + 1,
            "item":       r["item"],
            "base_price": _state["item_values"].get(r["item"], 0),
            "is_mystery": r["item"].lower() in mb_items,
            "value":      r["value"],
            "p_solo":     round(r["p_solo"] * 100, 1),
            "ev":         round(r["ev"], 2),
            "history":    r["history"],
        }
        if my_player:
            h = my_hist.get(r["item"])
            if h and h.get("my_picks", 0) > 0:
                row["my_tag"] = f"{h['my_wins']}W/{h['my_collide']}C in {h['my_picks']} picks"
        ranked.append(row)

    ev_label  = {"linear": "EV", "exp": "EV(^.57)", "log": "EV(log)"}.get(utility, "EV")
    util_desc = UTILITY_LABELS.get(utility, utility)
    return jsonify({"ranked": ranked, "ev_label": ev_label,
                    "utility_desc": util_desc, "utility": utility,
                    "multipliers": mults})


@app.route("/api/round/save", methods=["POST"])
def api_round_save():
    if not _pending.get("items"):
        return jsonify({"error": "No pending round."}), 400

    data      = request.get_json()
    entries   = data.get("results", [])
    aliases   = _state.get("name_aliases", {})
    new_round = {"items": {}}

    for entry in entries:
        item       = entry.get("item", "").strip()
        raw_players = [p.strip() for p in entry.get("players", []) if str(p).strip()]
        if not item or not raw_players:
            continue
        canon   = canonical_item(item, _state["item_values"]) or item
        players = [normalise_player(p, _player_profiles, aliases) for p in raw_players]
        new_round["items"][canon] = players

    if not new_round["items"]:
        return jsonify({"error": "No results entered."}), 400

    for item in _pending["items"]:
        if item not in new_round["items"]:
            new_round["items"][item] = []

    # Store multipliers in round data (only entries > 1)
    pending_mults = _pending.get("multipliers", {})
    round_mults   = {k: v for k, v in pending_mults.items() if v > 1}
    if round_mults:
        new_round["multipliers"] = round_mults

    mb_items = _mystery_box_items()
    winners = [
        {"item": item, "player": players[0],
         "value": _state["item_values"].get(item, 0) * pending_mults.get(item, 1),
         "is_mystery_box": item.lower() in mb_items}
        for item, players in new_round["items"].items() if len(players) == 1
    ]

    test_mode = bool(data.get("test_mode", False))
    if not test_mode:
        _state["rounds"].append(new_round)
        save_state(_state)
        _retrain()
    _pending.clear()

    return jsonify({"ok": True, "winners": winners,
                    "round_num": len(_state["rounds"]), "test_mode": test_mode})


# ---------------------------------------------------------------------------
# API — Statistics
# ---------------------------------------------------------------------------

@app.route("/api/stats/players")
def api_stats_players():
    # Tally total winnings per player from round history
    winnings = defaultdict(float)
    for rd in _state["rounds"]:
        rd_mults = rd.get("multipliers", {})
        for item, players in rd["items"].items():
            if len(players) == 1:
                base  = _state["item_values"].get(item, 0)
                mult  = rd_mults.get(item, 1)
                winnings[players[0].lower()] += base * mult

    rows = [
        {"name": name, "picks": p["picks"], "wins": p["wins"],
         "win_rate": round(p["win_rate"] * 100, 1),
         "contrarian": round(p.get("contrarian", 0), 2),
         "total_winnings": winnings.get(name.lower(), 0)}
        for name, p in _player_profiles.items() if p.get("picks", 0) >= 1
    ]
    rows.sort(key=lambda x: -x["win_rate"])
    return jsonify(rows)


@app.route("/api/stats/items")
def api_stats_items():
    item_stats = defaultdict(lambda: {"appearances": 0, "picked": 0, "total": 0, "solo": 0})
    for rd in _state["rounds"]:
        for item, players in rd["items"].items():
            s = item_stats[item]
            s["appearances"] += 1
            if players:
                s["picked"] += 1
                s["total"]  += len(players)
                if len(players) == 1:
                    s["solo"] += 1

    rows = []
    for item, price in _state["item_values"].items():
        s   = item_stats.get(item, {})
        pk  = s.get("picked", 0)
        rows.append({
            "item": item, "price": price,
            "appearances": s.get("appearances", 0), "picked": pk,
            "solo": s.get("solo", 0),
            "avg_pickers": round(s.get("total", 0) / pk, 1) if pk else 0,
        })
    rows.sort(key=lambda x: -x["price"])
    return jsonify(rows)


@app.route("/api/items/update_price", methods=["POST"])
def api_items_update_price():
    data  = request.get_json()
    item  = data.get("item", "").strip()
    price = data.get("price")
    if not item or item not in _state["item_values"]:
        return jsonify({"error": "Item not found."}), 404
    try:
        price = float(str(price).replace(",", ""))
        if price < 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid price."}), 400
    _state["item_values"][item] = price
    save_state(_state)
    return jsonify({"ok": True})


@app.route("/api/items/rename", methods=["POST"])
def api_items_rename():
    data     = request.get_json()
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    if not old_name or not new_name:
        return jsonify({"error": "Both names required."}), 400
    if old_name == new_name:
        return jsonify({"ok": True})
    if old_name not in _state["item_values"]:
        return jsonify({"error": f"Item '{old_name}' not found."}), 404
    canon_new = canonical_item(new_name, _state["item_values"])
    if canon_new and canon_new != old_name:
        return jsonify({"error": f"'{new_name}' already exists."}), 400
    price = _state["item_values"].pop(old_name)
    _state["item_values"][new_name] = price
    for rd in _state["rounds"]:
        if old_name in rd["items"]:
            rd["items"][new_name] = rd["items"].pop(old_name)
    save_state(_state)
    _retrain()
    return jsonify({"ok": True, "new_name": new_name})


@app.route("/api/stats/player/<path:name>")
def api_stats_player(name):
    canon = canonical_player(name, _player_profiles)
    if not canon:
        matches = [p for p in _player_profiles if name.lower() in p.lower()]
        if len(matches) == 1:
            canon = matches[0]
        elif len(matches) > 1:
            return jsonify({"error": f"Multiple matches: {', '.join(matches[:5])}"})
        else:
            return jsonify({"error": f"No player found matching '{name}'."})

    p_lower = canon.lower()
    rows = []
    rounds_played = wins = collisions = total_picks = 0
    for i, rd in enumerate(_state["rounds"]):
        participated = False
        rd_mults = rd.get("multipliers", {})
        for item, players in rd["items"].items():
            if any(p.lower() == p_lower for p in players):
                participated = True
                total_picks += 1
                won = len(players) == 1
                wins      += int(won)
                collisions += int(not won)
                base  = _state["item_values"].get(item, 0)
                mult  = rd_mults.get(item, 1)
                rows.append({"round": i + 1, "item": item, "won": won,
                             "others": len(players) - 1,
                             "value": base * mult, "mult": mult})
        if participated:
            rounds_played += 1

    return jsonify({"name": canon, "rounds_played": rounds_played, "picks": total_picks,
                    "wins": wins, "collisions": collisions,
                    "win_rate": round(wins / total_picks * 100, 1) if total_picks else 0,
                    "rows": rows})


@app.route("/api/stats/item/<path:name>")
def api_stats_item(name):
    canon = canonical_item(name, _state["item_values"])
    if not canon:
        all_items = set(_state["item_values"].keys())
        for rd in _state["rounds"]:
            all_items.update(rd["items"].keys())
        matches = sorted(i for i in all_items if name.lower() in i.lower())
        if len(matches) == 1:
            canon = matches[0]
        elif len(matches) > 1:
            return jsonify({"error": f"Multiple matches: {', '.join(matches[:5])}"})
        else:
            return jsonify({"error": f"No item found matching '{name}'."})

    tally = defaultdict(lambda: {"picks": 0, "wins": 0, "collisions": 0})
    appearances = picked_rounds = 0
    for rd in _state["rounds"]:
        players = rd["items"].get(canon)
        if players is None:
            continue
        appearances += 1
        if not players:
            continue
        picked_rounds += 1
        won = len(players) == 1
        for p in players:
            tally[p]["picks"] += 1
            if won:
                tally[p]["wins"] += 1
            else:
                tally[p]["collisions"] += 1

    rows = sorted(tally.items(), key=lambda x: (-x[1]["picks"], -x[1]["wins"]))
    return jsonify({
        "item": canon, "price": _state["item_values"].get(canon, 0),
        "appearances": appearances, "picked": picked_rounds,
        "skipped": appearances - picked_rounds,
        "rows": [{"player": p, **t,
                  "pick_rate": round(t["picks"] / appearances * 100, 1) if appearances else 0}
                 for p, t in rows],
    })


@app.route("/api/stats/search/players")
def api_search_players():
    q = request.args.get("q", "").lower()
    return jsonify(sorted(p for p in _player_profiles if q in p.lower())[:20])


@app.route("/api/stats/search/items")
def api_search_items():
    q = request.args.get("q", "").lower()
    all_items = set(_state["item_values"].keys())
    for rd in _state["rounds"]:
        all_items.update(rd["items"].keys())
    return jsonify(sorted(i for i in all_items if q in i.lower())[:20])


# ---------------------------------------------------------------------------
# API — My Stats
# ---------------------------------------------------------------------------

@app.route("/api/mystats")
def api_mystats():
    my_player = _state.get("my_player")
    if not my_player:
        return jsonify({"error": "No player name set. Go to Settings."})

    my_lower = my_player.lower()
    rows = []
    rounds_played = wins = collisions = total_picks = 0
    for i, rd in enumerate(_state["rounds"]):
        participated = False
        for item, players in rd["items"].items():
            if any(p.lower() == my_lower for p in players):
                participated = True
                total_picks += 1
                won = len(players) == 1
                wins       += int(won)
                collisions += int(not won)
                rows.append({"round": i + 1, "item": item, "won": won,
                             "others": len(players) - 1,
                             "value": _state["item_values"].get(item, 0)})
        if participated:
            rounds_played += 1

    return jsonify({"name": my_player, "rounds_played": rounds_played, "picks": total_picks,
                    "wins": wins, "collisions": collisions,
                    "win_rate": round(wins / total_picks * 100, 1) if total_picks else 0,
                    "rows": rows})


# ---------------------------------------------------------------------------
# API — Backfill multipliers after repricing
# ---------------------------------------------------------------------------

@app.route("/api/admin/backfill_multipliers", methods=["POST"])
def api_backfill_multipliers():
    """
    For every item now priced below 200,000, find all historical round appearances
    that have no existing multiplier and attach a multiplier high enough so that
    base_price * multiplier > 500,000.
    Call this after manually repricing items in the Item List.
    """
    changed_items = {}
    for item, price in _state["item_values"].items():
        if price >= 200_000:
            continue
        if price <= 0:
            continue
        mult = math.ceil(500_001 / price)
        applied = 0
        for rd in _state["rounds"]:
            if item not in rd["items"]:
                continue
            rd_mults = rd.setdefault("multipliers", {})
            if item in rd_mults:
                continue  # already has an explicit multiplier — don't overwrite
            rd_mults[item] = mult
            applied += 1
        if applied:
            changed_items[item] = {"multiplier": mult, "rounds_updated": applied,
                                   "price": price, "effective": price * mult}

    save_state(_state)
    return jsonify({"ok": True, "updated": changed_items,
                    "total_items": len(changed_items)})


# ---------------------------------------------------------------------------
# API — Mystery boxes
# ---------------------------------------------------------------------------

def _mystery_box_items():
    """Return the set of canonical mystery-box item names (lowercase)."""
    mb = _state.get("mystery_boxes", {})
    return {n.lower() for n in mb.get("items", [])}

def _mystery_box_sync():
    """Recompute expected_value from observations and update item prices."""
    mb = _state["mystery_boxes"]
    obs = mb.get("observations", [])
    ev = round(sum(obs) / len(obs)) if obs else None
    mb["expected_value"] = ev
    if ev is not None:
        for name in mb.get("items", []):
            canon = canonical_item(name, _state["item_values"])
            if canon:
                _state["item_values"][canon] = float(ev)


@app.route("/api/mystery_box/observe", methods=["POST"])
def api_mystery_box_observe():
    data  = request.get_json()
    value = data.get("value")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid value."}), 400
    if value <= 0:
        return jsonify({"error": "Value must be positive."}), 400

    mb = _state.setdefault("mystery_boxes", {"items": ["box", "white elephant gift"],
                                              "observations": [], "expected_value": None})
    mb.setdefault("observations", []).append(value)
    _mystery_box_sync()
    save_state(_state)
    _retrain()
    return jsonify({"ok": True, "observations": mb["observations"],
                    "expected_value": mb["expected_value"]})


@app.route("/api/mystery_box/config", methods=["GET"])
def api_mystery_box_config():
    mb = _state.get("mystery_boxes", {})
    return jsonify({
        "items":          mb.get("items", []),
        "observations":   mb.get("observations", []),
        "expected_value": mb.get("expected_value"),
    })


# ---------------------------------------------------------------------------
# API — Settings
# ---------------------------------------------------------------------------

@app.route("/api/settings")
def api_settings():
    return jsonify({
        "my_player":    _state.get("my_player", ""),
        "decay_factor": _state.get("decay_factor", DEFAULT_DECAY),
        "utility_mode": _state.get("utility_mode", DEFAULT_UTILITY),
        "aliases":      _state.get("name_aliases", {}),
    })


@app.route("/api/settings/player", methods=["POST"])
def api_settings_player():
    name = request.get_json().get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required."}), 400
    canon = canonical_player(name, _player_profiles)
    _state["my_player"] = canon if canon else name
    save_state(_state)
    return jsonify({"ok": True, "name": _state["my_player"]})


@app.route("/api/settings/decay", methods=["POST"])
def api_settings_decay():
    try:
        decay = float(request.get_json().get("decay", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value."}), 400
    if not (0.5 <= decay <= 1.0):
        return jsonify({"error": "Decay must be between 0.5 and 1.0."}), 400
    _state["decay_factor"] = decay
    save_state(_state)
    _retrain()
    return jsonify({"ok": True, "decay": decay})


@app.route("/api/settings/utility", methods=["POST"])
def api_settings_utility():
    utility = request.get_json().get("utility", "").strip()
    if utility not in UTILITY_LABELS:
        return jsonify({"error": "Choose linear, exp, or log."}), 400
    _state["utility_mode"] = utility
    save_state(_state)
    return jsonify({"ok": True, "utility": utility})


@app.route("/api/settings/merge", methods=["POST"])
def api_settings_merge():
    data      = request.get_json()
    from_name = data.get("from_name", "").strip()
    to_name   = data.get("to_name",   "").strip()

    from_canon  = canonical_player(from_name, _player_profiles)
    to_resolved = canonical_player(to_name,   _player_profiles)
    if not from_canon:
        return jsonify({"error": f"Player '{from_name}' not found."}), 400
    if not to_name.strip():
        return jsonify({"error": "Target name is empty."}), 400

    same_spelling = from_canon.lower() == to_name.lower()
    if same_spelling:
        # Pure recapitalisation — use the typed form as the new canonical spelling
        to_canon = to_name
    else:
        # Merge — resolve to existing canonical name, or use typed value if new
        to_canon = to_resolved or to_name

    # Recapitalise all historical occurrences
    old_lower = from_canon.lower()
    changed = 0
    for rd in _state["rounds"]:
        for players in rd["items"].values():
            for i, p in enumerate(players):
                if p.lower() == old_lower:
                    players[i] = to_canon
                    changed += 1
    if _state.get("my_player", "").lower() == old_lower:
        _state["my_player"] = to_canon

    aliases = _state.setdefault("name_aliases", {})
    # Update any existing aliases that point to the old capitalisation
    for key in list(aliases.keys()):
        if aliases[key].lower() == old_lower:
            aliases[key] = to_canon

    if same_spelling:
        # Pure recapitalisation — no alias needed (the normalised key is identical)
        pass
    else:
        # Different names — record alias so old name is auto-replaced in future input
        aliases[from_canon.lower()] = to_canon

    save_state(_state)
    _retrain()
    return jsonify({"ok": True, "changed": changed, "from": from_canon, "to": to_canon})


@app.route("/api/settings/alias/remove", methods=["POST"])
def api_settings_alias_remove():
    alias   = request.get_json().get("alias", "").strip().lower()
    aliases = _state.get("name_aliases", {})
    if alias not in aliases:
        return jsonify({"error": "Alias not found."}), 400
    del aliases[alias]
    save_state(_state)
    return jsonify({"ok": True})


@app.route("/api/items/merge", methods=["POST"])
def api_items_merge():
    data      = request.get_json()
    from_name = data.get("from_name", "").strip()
    to_name   = data.get("to_name",   "").strip()

    from_canon = canonical_item(from_name, _state["item_values"])
    to_canon   = canonical_item(to_name,   _state["item_values"])
    if not from_canon:
        return jsonify({"error": f"Item '{from_name}' not found."}), 400
    if not to_canon:
        return jsonify({"error": f"Item '{to_name}' not found."}), 400
    if from_canon.lower() == to_canon.lower():
        return jsonify({"error": "Same item — nothing to merge."}), 400

    del _state["item_values"][from_canon]
    changed = 0
    for rd in _state["rounds"]:
        if from_canon in rd["items"]:
            pickers = rd["items"].pop(from_canon)
            if to_canon not in rd["items"]:
                rd["items"][to_canon] = pickers
            else:
                # Both items appeared in the same round — combine picker lists
                merged = list({p for p in rd["items"][to_canon] + pickers})
                rd["items"][to_canon] = merged
            changed += 1

    save_state(_state)
    _retrain()
    return jsonify({"ok": True, "changed": changed, "from": from_canon, "to": to_canon})


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        rounds=len(_state["rounds"]),
        my_player=_state.get("my_player", ""),
    )



if __name__ == "__main__":
    print("  Nobody Wants It — Web UI")
    print("  Open http://localhost:5001 in your browser.")
    print("  Press Ctrl-C to stop.")
    app.run(host="127.0.0.1", port=5001, debug=False)
