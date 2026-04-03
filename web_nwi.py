"""
Nobody Wants It — Web UI
Run:   python web_nwi.py
Open:  http://localhost:5000
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict

from flask import Flask, request, jsonify, render_template_string

sys.path.insert(0, str(Path(__file__).parent))
from nobody_wants_it import (
    load_state, save_state, build_player_profiles, NWIModel,
    canonical_item, canonical_player, normalise_item, normalise_player,
    build_my_item_history, DEFAULT_DECAY, DEFAULT_UTILITY, UTILITY_LABELS,
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
    _model.train(_state["rounds"], _player_profiles, _state["item_values"],
                 decay=_state.get("decay_factor", DEFAULT_DECAY))

# Pending round being constructed across steps
_pending: dict = {}   # {participants, items}


def _retrain():
    global _player_profiles
    _player_profiles = build_player_profiles(_state["rounds"])
    _model.train(_state["rounds"], _player_profiles, _state["item_values"],
                 decay=_state.get("decay_factor", DEFAULT_DECAY))


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
    raw_items    = [l.strip() for l in data.get("items", "").splitlines() if l.strip()]
    items        = [normalise_item(i, _state["item_values"]) for i in raw_items]

    if not participants:
        return jsonify({"error": "Please enter at least one participant."}), 400
    if not items:
        return jsonify({"error": "Please enter at least one item."}), 400

    new_items = [i for i in items if canonical_item(i, _state["item_values"]) is None]
    _pending  = {"participants": participants, "items": items}
    return jsonify({"new_items": new_items, "participants": participants, "items": items})


@app.route("/api/round/prices", methods=["POST"])
def api_round_prices():
    """Step 2: receive prices (or merge targets) for new items."""
    data   = request.get_json()
    prices = data.get("prices", {})

    for item, raw_val in prices.items():
        raw_val = str(raw_val).strip()
        # Try to parse as a number
        try:
            price = float(raw_val.replace(",", ""))
            _state["item_values"][item] = price
        except ValueError:
            # Try to treat as a merge target (existing item name)
            canon = canonical_item(raw_val, _state["item_values"])
            if canon:
                # Replace all occurrences in pending items
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

    utility    = _state.get("utility_mode", DEFAULT_UTILITY)
    round_items = {i: _state["item_values"].get(i, 0) for i in _pending["items"]}
    results    = _model.score_items(round_items, _pending["participants"],
                                    _player_profiles, utility=utility)
    my_player  = _state.get("my_player")
    my_hist    = build_my_item_history(_state["rounds"], my_player) if my_player else {}

    ranked = []
    for r in results:
        row = {
            "rank":    len(ranked) + 1,
            "item":    r["item"],
            "value":   r["value"],
            "p_solo":  round(r["p_solo"] * 100, 1),
            "ev":      round(r["ev"], 2),
            "history": r["history"],
        }
        if my_player:
            h = my_hist.get(r["item"])
            if h and h.get("my_picks", 0) > 0:
                row["my_tag"] = f"{h['my_wins']}W/{h['my_collide']}C in {h['my_picks']} picks"
        ranked.append(row)

    ev_label  = {"linear": "EV", "sqrt": "EV(√)", "log": "EV(log)"}.get(utility, "EV")
    util_desc = UTILITY_LABELS.get(utility, utility)
    return jsonify({"ranked": ranked, "ev_label": ev_label,
                    "utility_desc": util_desc, "utility": utility})


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

    winners = [
        {"item": item, "player": players[0], "value": _state["item_values"].get(item, 0)}
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
    rows = [
        {"name": name, "picks": p["picks"], "wins": p["wins"],
         "win_rate": round(p["win_rate"] * 100, 1),
         "contrarian": round(p.get("contrarian", 0), 2)}
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
        for item, players in rd["items"].items():
            if any(p.lower() == p_lower for p in players):
                participated = True
                total_picks += 1
                won = len(players) == 1
                wins      += int(won)
                collisions += int(not won)
                rows.append({"round": i + 1, "item": item, "won": won,
                             "others": len(players) - 1,
                             "value": _state["item_values"].get(item, 0)})
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
        return jsonify({"error": "Choose linear, sqrt, or log."}), 400
    _state["utility_mode"] = utility
    save_state(_state)
    return jsonify({"ok": True, "utility": utility})


@app.route("/api/settings/merge", methods=["POST"])
def api_settings_merge():
    data      = request.get_json()
    from_name = data.get("from_name", "").strip()
    to_name   = data.get("to_name",   "").strip()

    from_canon = canonical_player(from_name, _player_profiles)
    to_canon   = canonical_player(to_name,   _player_profiles)
    if not from_canon:
        return jsonify({"error": f"Player '{from_name}' not found."}), 400
    if not to_canon:
        return jsonify({"error": f"Player '{to_name}' not found."}), 400
    if from_canon.lower() == to_canon.lower():
        return jsonify({"error": "Same player — nothing to merge."}), 400

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

    # Auto-record alias so future input of the old name maps to the canonical name
    _state.setdefault("name_aliases", {})[from_canon.lower()] = to_canon

    save_state(_state)
    _retrain()
    return jsonify({"ok": True, "changed": changed, "from": from_canon, "to": to_canon})


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
    return render_template_string(
        MAIN_HTML,
        rounds=len(_state["rounds"]),
        my_player=_state.get("my_player", ""),
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

MAIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Nobody Wants It</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d1117; color: #c9d1d9; font-family: system-ui, sans-serif;
           height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

    /* Topbar */
    .topbar { background: #161b22; border-bottom: 1px solid #30363d; padding: .5rem 1rem;
              display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
    .topbar-title { color: #58a6ff; font-weight: 600; font-size: 1rem; }
    .topbar-info  { font-size: .78rem; color: #8b949e; }

    /* Layout */
    .body    { display: flex; flex: 1; overflow: hidden; }
    .sidebar { width: 185px; background: #161b22; border-right: 1px solid #30363d;
               padding: .75rem .6rem; overflow-y: auto; flex-shrink: 0; }
    .main    { flex: 1; overflow-y: auto; padding: 1rem; }

    /* Sidebar nav */
    .nav-section { font-size: .7rem; color: #484f58; text-transform: uppercase;
                   letter-spacing: .06em; padding: .6rem .4rem .2rem; }
    .nav-btn { display: block; width: 100%; text-align: left; background: transparent;
               border: none; color: #8b949e; padding: .35rem .6rem; border-radius: 4px;
               cursor: pointer; font-size: .82rem; margin-bottom: 1px; }
    .nav-btn:hover  { background: #21262d; color: #c9d1d9; }
    .nav-btn.active { background: #21262d; color: #58a6ff; }

    /* Panels */
    .panel        { display: none; }
    .panel.active { display: block; }

    /* Cards */
    .card    { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
               padding: 1rem; margin-bottom: .75rem; }
    .card h6 { color: #e6edf3; font-size: .9rem; margin-bottom: .75rem; font-weight: 600; }
    .card-sub { font-size: .75rem; color: #8b949e; margin-bottom: .75rem; }

    /* Forms */
    .field       { display: flex; flex-direction: column; gap: .25rem; margin-bottom: .6rem; }
    .field label { font-size: .75rem; color: #8b949e; }
    input[type=text], input[type=number], textarea, select {
      padding: .35rem .5rem; background: #0d1117; border: 1px solid #30363d;
      border-radius: 4px; color: #c9d1d9; font-size: .83rem; outline: none; }
    input[type=text]:focus, input[type=number]:focus, textarea:focus, select:focus {
      border-color: #58a6ff; }
    textarea { width: 100%; resize: vertical; font-family: monospace; }
    input[type=text], input[type=number], select { width: 100%; }

    /* Buttons */
    .btn         { padding: .35rem .85rem; border: none; border-radius: 4px; font-size: .83rem; cursor: pointer; }
    .btn-primary { background: #238636; color: #fff; }
    .btn-primary:hover  { background: #2ea043; }
    .btn-secondary      { background: transparent; border: 1px solid #30363d; color: #8b949e; }
    .btn-secondary:hover { border-color: #8b949e; color: #c9d1d9; }
    .btn-danger         { background: #7a1f1f; color: #fca5a5; }
    .btn-danger:hover   { background: #991b1b; }
    .btn-sm { padding: .2rem .55rem; font-size: .75rem; }
    .btn-group { display: flex; gap: .5rem; flex-wrap: wrap; }

    /* Messages */
    .msg     { padding: .4rem .7rem; border-radius: 4px; font-size: .82rem; margin-bottom: .6rem; }
    .msg-ok  { background: #0c2d1a; color: #4ade80; border: 1px solid #0c6b3a; }
    .msg-err { background: #2d0c0c; color: #f87171; border: 1px solid #6b1010; }
    .hidden  { display: none !important; }

    /* Tables */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .8rem; }
    thead th { position: sticky; top: 0; background: #1c2128; color: #8b949e; font-weight: normal;
               padding: .3rem .6rem; border-bottom: 1px solid #30363d;
               text-align: left; white-space: nowrap; z-index: 1; }
    tbody td { padding: .25rem .6rem; border-bottom: 1px solid #21262d; }
    tbody tr:hover td { background: #21262d; }
    .td-right { text-align: right; }
    .td-mono  { font-family: monospace; }
    .sortable { cursor: pointer; user-select: none; }
    .sortable:hover { color: #c9d1d9; }
    .sort-ind { margin-left: .3rem; font-size: .7rem; opacity: .6; }
    .price-inp { width: 90px; background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
                 border-radius: 3px; padding: 1px 4px; font-family: monospace; font-size: .8rem;
                 text-align: right; }
    .price-inp:focus { border-color: #388bfd; outline: none; }
    td:has(> .editable-cell) { padding: 0; }
    .editable-cell { background: transparent; border: 1px solid transparent; color: #c9d1d9;
                     border-radius: 3px; padding: 3px 6px; font-size: .8rem; cursor: pointer;
                     font-family: inherit; }
    .editable-cell:hover { border-color: #30363d; }
    .editable-cell:focus { background: #161b22; border-color: #388bfd; outline: none; cursor: text; }
    .editable-cell.ec-name  { width: 100%; box-sizing: border-box; min-width: 0;
                               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .editable-cell.ec-name:focus { white-space: normal; overflow: visible; }
    .editable-cell.ec-price { text-align: right; font-family: monospace; width: 120px; }
    .toggle-wrap { display:flex; align-items:center; gap:.45rem; font-size:.8rem; color:#8b949e; cursor:pointer; }
    .toggle-wrap input { display:none; }
    .toggle-track { width:34px; height:18px; background:#30363d; border-radius:9px; position:relative;
                    transition:background .2s; flex-shrink:0; }
    .toggle-track::after { content:''; position:absolute; top:3px; left:3px; width:12px; height:12px;
                           background:#8b949e; border-radius:50%; transition:transform .2s, background .2s; }
    .toggle-wrap input:checked + .toggle-track { background:#b45309; }
    .toggle-wrap input:checked + .toggle-track::after { transform:translateX(16px); background:#f0883e; }

    /* Recommendation table */
    .rec-top td     { color: #fbbf24; }
    .psolo-hi       { color: #4ade80; }
    .psolo-mid      { color: #fbbf24; }
    .psolo-lo       { color: #f87171; }
    .you-tag        { font-size: .72rem; color: #58a6ff; margin-left: .4rem; }
    .rec-footer     { font-size: .73rem; color: #8b949e; margin-top: .4rem; }

    /* Results table */
    .res-input { width: 100%; background: #0d1117; border: 1px solid #21262d;
                 border-radius: 3px; color: #c9d1d9; font-size: .8rem;
                 padding: .2rem .4rem; outline: none; }
    .res-input:focus { border-color: #58a6ff; }

    /* Winners */
    .winner-row { display: flex; align-items: center; gap: .5rem; padding: .4rem .6rem;
                  background: #0c2d1a; border: 1px solid #0c6b3a; border-radius: 4px;
                  margin-bottom: .4rem; font-size: .85rem; color: #4ade80; }
    .winner-val { margin-left: auto; color: #8b949e; font-size: .78rem; }

    /* Stat boxes */
    .stat-row  { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: .75rem; }
    .stat-box  { background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
                 padding: .5rem .85rem; text-align: center; min-width: 70px; }
    .stat-val  { font-size: 1.2rem; font-weight: 600; color: #e6edf3; }
    .stat-lbl  { font-size: .7rem; color: #8b949e; margin-top: .1rem; }

    /* Search suggestions */
    .suggestions { margin-bottom: .5rem; }
    .search-row  { display: flex; gap: .5rem; margin-bottom: .75rem; }
    .search-row input { flex: 1; }
    /* Detail expand rows */
    .detail-row > td { background: #0d1117 !important; border-bottom: 2px solid #30363d; padding: .6rem .8rem; }
    .detail-wrap .stat-row { margin-bottom: .5rem; }
    .detail-wrap table { margin-top: .4rem; }
    .detail-btn { padding: .1rem .45rem; font-size: .75rem; cursor: pointer;
                  background: transparent; border: 1px solid #30363d; color: #8b949e;
                  border-radius: 3px; white-space: nowrap; }
    .detail-btn:hover { border-color: #8b949e; color: #c9d1d9; }

    /* Settings */
    .settings-sep { font-size: .75rem; color: #484f58; text-transform: uppercase;
                    letter-spacing: .05em; border-bottom: 1px solid #30363d;
                    padding-bottom: .3rem; margin: 1rem 0 .6rem; }
  </style>
</head>
<body>

<div class="topbar">
  <span class="topbar-title">Nobody Wants It — Advisor</span>
  <span class="topbar-info" id="topbar-info">
    {{ rounds }} round{{ 's' if rounds != 1 else '' }}{% if my_player %} · {{ my_player }}{% endif %}
  </span>
</div>

<div class="body">
  <div class="sidebar">
    <div class="nav-section">Actions</div>
    <button class="nav-btn active" onclick="showPanel('round', this)">▶ Play a Round</button>
    <div class="nav-section">Statistics</div>
    <button class="nav-btn" onclick="showPanel('players', this)">Player Leaderboard</button>
    <button class="nav-btn" onclick="showPanel('items-list', this)">Item List</button>
    <div class="nav-section">Personal</div>
    <button class="nav-btn" onclick="showPanel('mystats', this)">My Stats</button>
    <div class="nav-section">Config</div>
    <button class="nav-btn" onclick="showPanel('settings', this)">Settings</button>
  </div>

  <div class="main">

    <!-- ─── ROUND PANEL ─── -->
    <div class="panel active" id="panel-round">
      <div id="round-msg"></div>

      <!-- Step 1: Setup -->
      <div class="card" id="step-setup">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem">
          <h6 style="margin:0">Play a Round</h6>
          <label class="toggle-wrap">
            <input type="checkbox" id="chk-test-mode">
            <span class="toggle-track"></span>
            Test mode
          </label>
        </div>
        <div class="field">
          <label>Participants (comma-separated)</label>
          <input type="text" id="inp-participants" placeholder="Alice, Bob, Carol...">
        </div>
        <div class="field">
          <label>Items available this round (one per line)</label>
          <textarea id="inp-items" rows="10" placeholder="spice melange&#10;Covert Ops for Kids&#10;tiny plastic sword&#10;..."></textarea>
        </div>
        <div class="btn-group">
          <button class="btn btn-primary" onclick="submitSetup()">Continue →</button>
        </div>
      </div>

      <!-- Step 2: Prices for new items -->
      <div class="card hidden" id="step-prices">
        <h6>New Items — Enter Prices</h6>
        <div class="card-sub">Enter a price in meat. Or type an existing item name to merge (e.g. fix a typo).</div>
        <div id="price-fields"></div>
        <div class="btn-group" style="margin-top:.5rem">
          <button class="btn btn-primary" onclick="submitPrices()">Save Prices →</button>
          <button class="btn btn-secondary" onclick="resetRound()">Cancel</button>
        </div>
      </div>

      <!-- Step 3: Recommendation -->
      <div class="card hidden" id="step-recommend">
        <h6>Recommendation</h6>
        <div class="btn-group" style="margin-bottom:.75rem">
          <button class="btn btn-secondary" onclick="getRecommendation()">Get Recommendation</button>
        </div>
        <div id="rec-output" class="hidden">
          <div class="tbl-wrap">
            <table>
              <thead><tr>
                <th>#</th><th>Item</th>
                <th class="td-right">Price</th>
                <th class="td-right">P(solo)</th>
                <th class="td-right" id="ev-col-hdr">EV</th>
                <th>Notes</th>
              </tr></thead>
              <tbody id="rec-tbody"></tbody>
            </table>
          </div>
          <div class="rec-footer" id="rec-footer"></div>
        </div>
      </div>

      <!-- Step 3b: Results -->
      <div class="card hidden" id="step-results">
        <h6>Enter Results</h6>
        <div class="card-sub">Fill in who picked each item. Leave blank for items that were skipped.</div>
        <div class="tbl-wrap" style="margin-bottom:.75rem">
          <table>
            <thead><tr>
              <th>Item</th>
              <th>Players who picked it (comma-separated)</th>
            </tr></thead>
            <tbody id="results-tbody"></tbody>
          </table>
        </div>
        <div class="btn-group">
          <button class="btn btn-primary" onclick="saveRound()">Save Round</button>
          <button class="btn btn-secondary" onclick="resetRound()">Start Over</button>
        </div>
      </div>

      <!-- Step 4: Winners -->
      <div class="card hidden" id="step-winners">
        <h6 id="winners-heading">Round Saved!</h6>
        <div id="winners-display"></div>
        <div class="btn-group" style="margin-top:.75rem">
          <button class="btn btn-primary" onclick="resetRound()">Play Another Round</button>
        </div>
      </div>
    </div>

    <!-- ─── PLAYER LEADERBOARD ─── -->
    <div class="panel" id="panel-players">
      <div class="card">
        <h6>Player Leaderboard</h6>
        <input type="text" id="players-search" placeholder="Filter by name..."
          oninput="filterPlayers(this.value)"
          style="margin-bottom:.6rem;max-width:240px;background:#161b22;border:1px solid #30363d;
                 color:#c9d1d9;border-radius:4px;padding:.25rem .5rem;font-size:.85rem;width:100%">
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th class="sortable" onclick="sortPlayers('name')">Player<span class="sort-ind" id="psort-name"></span></th>
              <th class="td-right sortable" onclick="sortPlayers('picks')">Picks<span class="sort-ind" id="psort-picks"></span></th>
              <th class="td-right sortable" onclick="sortPlayers('wins')">Wins<span class="sort-ind" id="psort-wins"> ▼</span></th>
              <th class="td-right sortable" onclick="sortPlayers('win_rate')">Win%<span class="sort-ind" id="psort-win_rate"></span></th>
              <th class="td-right sortable" onclick="sortPlayers('contrarian')">Contrarian<span class="sort-ind" id="psort-contrarian"></span></th>
              <th></th>
            </tr></thead>
            <tbody id="players-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ─── ITEM LIST ─── -->
    <div class="panel" id="panel-items-list">
      <div class="card">
        <h6>Item List</h6>
        <div id="items-list-msg"></div>
        <input type="text" id="items-search" placeholder="Filter by name..."
          oninput="filterItems(this.value)"
          style="margin-bottom:.6rem;max-width:240px;background:#161b22;border:1px solid #30363d;
                 color:#c9d1d9;border-radius:4px;padding:.25rem .5rem;font-size:.85rem;width:100%">
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th style="max-width:200px" class="sortable" onclick="sortItems('item')">Item<span class="sort-ind" id="isort-item"></span></th>
              <th class="td-right sortable" onclick="sortItems('price')">Price<span class="sort-ind" id="isort-price"> ▼</span></th>
              <th class="td-right sortable" onclick="sortItems('appearances')">App.<span class="sort-ind" id="isort-appearances"></span></th>
              <th class="td-right sortable" onclick="sortItems('picked')">Picked<span class="sort-ind" id="isort-picked"></span></th>
              <th class="td-right sortable" onclick="sortItems('solo')">Solo<span class="sort-ind" id="isort-solo"></span></th>
              <th class="td-right sortable" onclick="sortItems('avg_pickers')">Avg Pickers<span class="sort-ind" id="isort-avg_pickers"></span></th>
              <th></th>
            </tr></thead>
            <tbody id="items-list-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ─── MY STATS ─── -->
    <div class="panel" id="panel-mystats">
      <div class="card">
        <h6>My Stats</h6>
        <div id="mystats-msg"></div>
        <div id="mystats-content" class="hidden">
          <div class="stat-row" id="mystats-stat-row"></div>
          <div class="tbl-wrap">
            <table>
              <thead><tr>
                <th>Rd</th><th>Item</th><th>Outcome</th>
                <th class="td-right">Others</th><th class="td-right">Value</th>
              </tr></thead>
              <tbody id="mystats-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- ─── SETTINGS ─── -->
    <div class="panel" id="panel-settings">
      <div class="card">
        <h6>Settings</h6>
        <div id="settings-msg"></div>

        <div class="settings-sep">My Player Name</div>
        <div class="field">
          <label>Name as it appears in round results</label>
          <input type="text" id="inp-my-player" placeholder="Your in-game name" style="max-width:260px">
        </div>
        <button class="btn btn-primary btn-sm" onclick="savePlayerName()">Save</button>

        <div class="settings-sep">Utility Mode</div>
        <div class="field" style="max-width:420px">
          <label>Scoring function used for recommendations</label>
          <select id="sel-utility">
            <option value="linear">linear — EV = price × P(solo)</option>
            <option value="sqrt">sqrt — EV = √price × P(solo)  [risk-adjusted]</option>
            <option value="log">log — EV = log(price) × P(solo)  [Kelly-style]</option>
          </select>
        </div>
        <button class="btn btn-primary btn-sm" onclick="saveUtility()">Save</button>

        <div class="settings-sep">Recency Decay Factor</div>
        <div class="field">
          <label>0.8 = older rounds count less · 1.0 = all rounds weighted equally</label>
          <input type="number" id="inp-decay" min="0.5" max="1.0" step="0.05" style="max-width:100px">
        </div>
        <button class="btn btn-primary btn-sm" onclick="saveDecay()">Save &amp; Retrain</button>

        <div class="settings-sep">Merge Player Names</div>
        <div class="card-sub" style="margin-bottom:.5rem">Rewrites all historical round data and records an automatic alias so the old name maps to the new one in future rounds.</div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:flex-end;margin-bottom:.6rem">
          <div class="field" style="flex:1;min-width:120px;margin-bottom:0">
            <label>Merge FROM (name to remove)</label>
            <input type="text" id="inp-merge-from" placeholder="Old name">
          </div>
          <div style="color:#8b949e;padding-bottom:.4rem">→</div>
          <div class="field" style="flex:1;min-width:120px;margin-bottom:0">
            <label>Merge INTO (name to keep)</label>
            <input type="text" id="inp-merge-into" placeholder="Canonical name">
          </div>
        </div>
        <button class="btn btn-danger btn-sm" onclick="mergePlayers()">Merge</button>

        <div class="settings-sep">Merge Items</div>
        <div class="card-sub" style="margin-bottom:.5rem">Combines two items into one, rewriting all historical round data. The merged item is removed from the item list.</div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:flex-end;margin-bottom:.6rem">
          <div class="field" style="flex:1;min-width:120px;margin-bottom:0">
            <label>Merge FROM (item to remove)</label>
            <input type="text" id="inp-item-merge-from" placeholder="Old item name">
          </div>
          <div style="color:#8b949e;padding-bottom:.4rem">→</div>
          <div class="field" style="flex:1;min-width:120px;margin-bottom:0">
            <label>Merge INTO (item to keep)</label>
            <input type="text" id="inp-item-merge-into" placeholder="Canonical item name">
          </div>
        </div>
        <button class="btn btn-danger btn-sm" onclick="mergeItems()">Merge</button>
      </div>
    </div>

  </div><!-- .main -->
</div><!-- .body -->

<script>
// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function fmt(n) { return Number(n).toLocaleString(); }
function pct(n) { return n.toFixed(1) + '%'; }

function showMsg(elId, text, type) {
  document.getElementById(elId).innerHTML =
    '<div class="msg msg-' + (type || 'ok') + '">' + text + '</div>';
}
function clearMsg(elId) { document.getElementById(elId).innerHTML = ''; }

async function post(url, body) {
  const r = await fetch(url, {method:'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  return r.json();
}

function statBox(val, lbl, color) {
  return '<div class="stat-box"><div class="stat-val"' +
    (color ? ' style="color:' + color + '"' : '') + '>' + val + '</div>' +
    '<div class="stat-lbl">' + lbl + '</div></div>';
}

function personalRows(tbody, rows) {
  tbody.innerHTML = '';
  rows.forEach(function(r) {
    var tr = document.createElement('tr');
    var style = r.won ? 'color:#4ade80' : 'color:#f87171';
    var outcome = r.won ? '🏆 Won' : '💥 Collision';
    tr.innerHTML = '<td>' + r.round + '</td><td>' + esc(r.item) + '</td>' +
      '<td style="' + style + '">' + outcome + '</td>' +
      '<td class="td-right">' + r.others + '</td>' +
      '<td class="td-right td-mono">' + fmt(r.value) + '</td>';
    tbody.appendChild(tr);
  });
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ---------------------------------------------------------------------------
// Panel switching
// ---------------------------------------------------------------------------
var panelLoaders = {};

function showPanel(panelId, btn) {
  document.querySelectorAll('.panel').forEach(function(p){ p.classList.remove('active'); });
  document.querySelectorAll('.nav-btn').forEach(function(b){ b.classList.remove('active'); });
  document.getElementById('panel-' + panelId).classList.add('active');
  if (btn) btn.classList.add('active');
  if (panelLoaders[panelId]) panelLoaders[panelId]();
}

// ---------------------------------------------------------------------------
// Round wizard
// ---------------------------------------------------------------------------
var roundItems = [];

function resetRound() {
  roundItems = [];
  document.getElementById('inp-participants').value = '';
  document.getElementById('inp-items').value = '';
  ['step-prices','step-recommend','step-results','step-winners'].forEach(function(id) {
    document.getElementById(id).classList.add('hidden');
  });
  document.getElementById('step-setup').classList.remove('hidden');
  document.getElementById('rec-output').classList.add('hidden');
  clearMsg('round-msg');
}

async function submitSetup() {
  var participants = document.getElementById('inp-participants').value.trim();
  var items = document.getElementById('inp-items').value.trim();
  if (!participants || !items) {
    showMsg('round-msg', 'Please enter participants and items.', 'err'); return;
  }
  var data = await post('/api/round/start', {participants: participants, items: items});
  if (data.error) { showMsg('round-msg', esc(data.error), 'err'); return; }
  clearMsg('round-msg');
  roundItems = data.items;

  if (data.new_items && data.new_items.length > 0) {
    buildPriceFields(data.new_items);
    document.getElementById('step-setup').classList.add('hidden');
    document.getElementById('step-prices').classList.remove('hidden');
  } else {
    document.getElementById('step-setup').classList.add('hidden');
    showRoundStep3();
  }
}

function buildPriceFields(newItems) {
  var container = document.getElementById('price-fields');
  container.innerHTML = '';
  newItems.forEach(function(item) {
    var d = document.createElement('div');
    d.className = 'field';
    d.innerHTML = '<label>' + esc(item) + '</label>' +
      '<input type="text" class="price-input" data-item="' + esc(item) + '" placeholder="Price in meat">';
    container.appendChild(d);
  });
}

async function submitPrices() {
  var prices = {};
  var ok = true;
  document.querySelectorAll('.price-input').forEach(function(inp) {
    var v = inp.value.trim();
    if (!v) { ok = false; }
    else { prices[inp.dataset.item] = v; }
  });
  if (!ok) { showMsg('round-msg', 'Please fill in all prices.', 'err'); return; }

  var data = await post('/api/round/prices', {prices: prices});
  if (data.error) { showMsg('round-msg', esc(data.error), 'err'); return; }
  roundItems = data.items;
  clearMsg('round-msg');
  document.getElementById('step-prices').classList.add('hidden');
  showRoundStep3();
}

function showRoundStep3() {
  document.getElementById('step-recommend').classList.remove('hidden');
  document.getElementById('step-results').classList.remove('hidden');
  buildResultsTable();
}

function buildResultsTable() {
  var tbody = document.getElementById('results-tbody');
  tbody.innerHTML = '';
  roundItems.forEach(function(item) {
    var tr = document.createElement('tr');
    tr.className = 'result-row';
    tr.dataset.item = item;
    tr.innerHTML = '<td style="white-space:nowrap;padding:.25rem .6rem;border-bottom:1px solid #21262d">' +
      esc(item) + '</td>' +
      '<td style="padding:.2rem .4rem;border-bottom:1px solid #21262d">' +
      '<input type="text" class="res-input result-players" placeholder="player1, player2 … (blank = skipped)"></td>';
    tbody.appendChild(tr);
  });
}

async function getRecommendation() {
  var data = await post('/api/round/recommend', {});
  if (data.error) { showMsg('round-msg', esc(data.error), 'err'); return; }

  document.getElementById('ev-col-hdr').textContent = data.ev_label;
  var tbody = document.getElementById('rec-tbody');
  tbody.innerHTML = '';
  data.ranked.forEach(function(r, i) {
    var tr = document.createElement('tr');
    if (i === 0) tr.className = 'rec-top';
    var psClass = r.p_solo >= 70 ? 'psolo-hi' : r.p_solo >= 40 ? 'psolo-mid' : 'psolo-lo';
    var star = i === 0 ? ' ★' : '';
    var youTag = r.my_tag ? '<span class="you-tag">[YOU: ' + esc(r.my_tag) + ']</span>' : '';
    var evStr = data.utility === 'linear' ? fmt(Math.round(r.ev)) : r.ev.toFixed(2);
    tr.innerHTML =
      '<td>' + r.rank + star + '</td>' +
      '<td>' + esc(r.item) + youTag + '</td>' +
      '<td class="td-right td-mono price-cell"></td>' +
      '<td class="td-right td-mono ' + psClass + '">' + pct(r.p_solo) + '</td>' +
      '<td class="td-right td-mono">' + evStr + '</td>' +
      '<td style="color:#8b949e;font-size:.75rem">' + esc(r.history) + '</td>';
    var priceInp = document.createElement('input');
    priceInp.className = 'price-inp';
    priceInp.value = r.value;
    priceInp.dataset.item = r.item;
    priceInp.dataset.orig = r.value;
    priceInp.addEventListener('blur', function() { updatePrice(this); });
    priceInp.addEventListener('keydown', function(e) { if (e.key === 'Enter') this.blur(); });
    tr.querySelector('.price-cell').appendChild(priceInp);
    tbody.appendChild(tr);
  });

  document.getElementById('rec-footer').textContent =
    '★ = top pick  |  ' + data.utility_desc + '  |  YOU: W=won, C=collision';
  document.getElementById('rec-output').classList.remove('hidden');
}

async function updatePrice(inp) {
  var raw   = inp.value.replace(/,/g, '').trim();
  var price = parseFloat(raw);
  if (isNaN(price) || price < 0) { inp.value = inp.dataset.orig; return; }
  if (price === parseFloat(inp.dataset.orig)) return;  // unchanged
  var data = await post('/api/round/update_price', {item: inp.dataset.item, price: price});
  if (data.error) { showMsg('round-msg', esc(data.error), 'err'); inp.value = inp.dataset.orig; return; }
  getRecommendation();  // refresh rankings with new price
}

async function saveRound() {
  var testMode = document.getElementById('chk-test-mode').checked;
  var results = [];
  document.querySelectorAll('.result-row').forEach(function(row) {
    var val = row.querySelector('.result-players').value.trim();
    if (val) {
      var players = val.split(',').map(function(p){ return p.trim(); }).filter(Boolean);
      if (players.length) results.push({item: row.dataset.item, players: players});
    }
  });

  var data = await post('/api/round/save', {results: results, test_mode: testMode});
  if (data.error) { showMsg('round-msg', esc(data.error), 'err'); return; }

  document.getElementById('winners-heading').textContent =
    testMode ? 'Test Round Complete (not saved)' : 'Round Saved!';

  var display = document.getElementById('winners-display');
  var prefix = testMode ? '<div class="card-sub" style="color:#f0883e;margin-bottom:.5rem">Test mode — this round was not recorded.</div>' : '';
  if (!data.winners.length) {
    display.innerHTML = prefix + '<div class="card-sub">No solo winners this round.</div>';
  } else {
    display.innerHTML = prefix + data.winners.map(function(w) {
      return '<div class="winner-row">🏆 <strong>' + esc(w.player) +
        '</strong> won <strong>' + esc(w.item) + '</strong>' +
        '<span class="winner-val">' + fmt(w.value) + ' meat</span></div>';
    }).join('');
  }

  // Only update topbar round count when actually saved
  if (!testMode) {
    var info = document.getElementById('topbar-info');
    var mp = info.textContent.indexOf('·');
    var suffix = mp !== -1 ? ' ' + info.textContent.substring(mp) : '';
    info.textContent = data.round_num + ' round' + (data.round_num !== 1 ? 's' : '') + suffix;
  }

  document.getElementById('step-recommend').classList.add('hidden');
  document.getElementById('step-results').classList.add('hidden');
  document.getElementById('step-winners').classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Player Leaderboard (sortable)
// ---------------------------------------------------------------------------
var _playersData = [];
var _playersSort = {col: 'wins', dir: -1};
var _playerFilter = '';
var _openPlayerDetail = null;

function filterPlayers(val) {
  _playerFilter = val.toLowerCase();
  renderPlayers();
}

function renderPlayers() {
  var col = _playersSort.col;
  var dir = _playersSort.dir;
  var data = _playerFilter
    ? _playersData.filter(function(r) { return r.name.toLowerCase().includes(_playerFilter); })
    : _playersData;
  var sorted = data.slice().sort(function(a, b) {
    var av = a[col], bv = b[col];
    if (typeof av === 'string') return dir * av.localeCompare(bv);
    return dir * (av - bv);
  });
  _openPlayerDetail = null;
  var tbody = document.getElementById('players-tbody');
  tbody.innerHTML = '';
  sorted.forEach(function(r) {
    var tr = document.createElement('tr');
    tr.innerHTML = '<td>' + esc(r.name) + '</td>' +
      '<td class="td-right">' + r.picks + '</td>' +
      '<td class="td-right">' + r.wins  + '</td>' +
      '<td class="td-right">' + pct(r.win_rate) + '</td>' +
      '<td class="td-right">' + r.contrarian.toFixed(2) + '</td>' +
      '<td class="td-right"></td>';
    var btn = document.createElement('button');
    btn.className = 'detail-btn';
    btn.textContent = 'Details';
    btn.dataset.name = r.name;
    btn.addEventListener('click', function() { togglePlayerDetail(this.dataset.name, this); });
    tr.lastElementChild.appendChild(btn);
    tbody.appendChild(tr);
  });
  ['name','picks','wins','win_rate','contrarian'].forEach(function(c) {
    var el = document.getElementById('psort-' + c);
    if (el) el.textContent = c === col ? (dir === -1 ? ' ▼' : ' ▲') : '';
  });
}

function sortPlayers(col) {
  if (_playersSort.col === col) {
    _playersSort.dir *= -1;
  } else {
    _playersSort.col = col;
    _playersSort.dir = col === 'name' ? 1 : -1;
  }
  renderPlayers();
}

async function togglePlayerDetail(name, btn) {
  var tr = btn.closest('tr');
  if (_openPlayerDetail === name) {
    tr.nextSibling.remove();
    _openPlayerDetail = null;
    btn.textContent = 'Details';
    return;
  }
  var prev = document.querySelector('#players-tbody .detail-row');
  if (prev) {
    prev.previousSibling.querySelector('.detail-btn').textContent = 'Details';
    prev.remove();
  }
  _openPlayerDetail = name;
  btn.textContent = 'Close';
  var detailTr = document.createElement('tr');
  detailTr.className = 'detail-row';
  detailTr.innerHTML = '<td colspan="6"><div class="detail-wrap"><span style="color:#8b949e;font-size:.8rem">Loading…</span></div></td>';
  tr.parentNode.insertBefore(detailTr, tr.nextSibling);
  var data = await (await fetch('/api/stats/player/' + encodeURIComponent(name))).json();
  if (data.error) {
    detailTr.querySelector('.detail-wrap').innerHTML = '<span style="color:#f87171">' + esc(data.error) + '</span>';
    return;
  }
  var wrap = detailTr.querySelector('.detail-wrap');
  var statRow = document.createElement('div');
  statRow.className = 'stat-row';
  statRow.innerHTML = statBox(data.rounds_played, 'Rounds') + statBox(data.picks, 'Picks') +
    statBox(data.wins, 'Wins', '#4ade80') + statBox(data.collisions, 'Collisions', '#f87171') +
    statBox(pct(data.win_rate), 'Win Rate');
  wrap.innerHTML = '';
  wrap.appendChild(statRow);
  var innerTbody = document.createElement('tbody');
  personalRows(innerTbody, data.rows);
  var innerTable = document.createElement('table');
  innerTable.innerHTML = '<thead><tr><th>Rd</th><th>Item</th><th>Outcome</th><th class="td-right">Others</th><th class="td-right">Value</th></tr></thead>';
  innerTable.appendChild(innerTbody);
  wrap.appendChild(innerTable);
}

panelLoaders['players'] = async function() {
  _playersData = await (await fetch('/api/stats/players')).json();
  _playersSort = {col: 'wins', dir: -1};
  _playerFilter = '';
  var searchEl = document.getElementById('players-search');
  if (searchEl) searchEl.value = '';
  renderPlayers();
};

// ---------------------------------------------------------------------------
// Item List
// ---------------------------------------------------------------------------
var _itemsData = [];
var _itemsSort = {col: 'price', dir: -1};
var _itemFilter = '';
var _openItemDetail = null;

function filterItems(val) {
  _itemFilter = val.toLowerCase();
  renderItems();
}

function renderItems() {
  var col = _itemsSort.col;
  var dir = _itemsSort.dir;
  var data = _itemFilter
    ? _itemsData.filter(function(r) { return r.item.toLowerCase().includes(_itemFilter); })
    : _itemsData;
  var sorted = data.slice().sort(function(a, b) {
    var av = a[col], bv = b[col];
    if (typeof av === 'string') return dir * av.localeCompare(bv);
    return dir * (av - bv);
  });
  _openItemDetail = null;
  var tbody = document.getElementById('items-list-tbody');
  tbody.innerHTML = '';
  sorted.forEach(function(r) {
    var tr = document.createElement('tr');
    tr.innerHTML =
      '<td class="item-name-cell"></td>' +
      '<td class="td-right td-mono price-cell"></td>' +
      '<td class="td-right">' + r.appearances + '</td>' +
      '<td class="td-right">' + r.picked + '</td>' +
      '<td class="td-right">' + r.solo + '</td>' +
      '<td class="td-right">' + (r.avg_pickers > 0 ? r.avg_pickers.toFixed(1) : '—') + '</td>' +
      '<td class="td-right"></td>';

    var nameInp = document.createElement('input');
    nameInp.className = 'editable-cell ec-name';
    nameInp.value = r.item;
    nameInp.dataset.orig = r.item;
    nameInp.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') this.blur();
      else if (e.key === 'Escape') { this.value = this.dataset.orig; this.blur(); }
    });
    nameInp.addEventListener('blur', function() { renameItem(this); });
    tr.querySelector('.item-name-cell').appendChild(nameInp);

    var priceInp = document.createElement('input');
    priceInp.className = 'editable-cell ec-price';
    priceInp.value = fmt(r.price);
    priceInp.dataset.item = r.item;
    priceInp.dataset.orig = r.price;
    priceInp.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') this.blur();
      else if (e.key === 'Escape') { this.value = fmt(parseFloat(this.dataset.orig)); this.blur(); }
    });
    priceInp.addEventListener('blur', function() { updateItemPrice(this); });
    tr.querySelector('.price-cell').appendChild(priceInp);

    var btn = document.createElement('button');
    btn.className = 'detail-btn';
    btn.textContent = 'Details';
    btn.dataset.item = r.item;
    btn.addEventListener('click', function() { toggleItemDetail(this.dataset.item, this); });
    tr.lastElementChild.appendChild(btn);

    tbody.appendChild(tr);
  });

  ['item','price','appearances','picked','solo','avg_pickers'].forEach(function(c) {
    var el = document.getElementById('isort-' + c);
    if (el) el.textContent = c === col ? (dir === -1 ? ' ▼' : ' ▲') : '';
  });
}

function sortItems(col) {
  if (_itemsSort.col === col) {
    _itemsSort.dir *= -1;
  } else {
    _itemsSort.col = col;
    _itemsSort.dir = col === 'item' ? 1 : -1;
  }
  renderItems();
}

async function toggleItemDetail(itemName, btn) {
  var tr = btn.closest('tr');
  if (_openItemDetail === itemName) {
    tr.nextSibling.remove();
    _openItemDetail = null;
    btn.textContent = 'Details';
    return;
  }
  var prev = document.querySelector('#items-list-tbody .detail-row');
  if (prev) {
    prev.previousSibling.querySelector('.detail-btn').textContent = 'Details';
    prev.remove();
  }
  _openItemDetail = itemName;
  btn.textContent = 'Close';
  var detailTr = document.createElement('tr');
  detailTr.className = 'detail-row';
  detailTr.innerHTML = '<td colspan="7"><div class="detail-wrap"><span style="color:#8b949e;font-size:.8rem">Loading…</span></div></td>';
  tr.parentNode.insertBefore(detailTr, tr.nextSibling);
  var data = await (await fetch('/api/stats/item/' + encodeURIComponent(itemName))).json();
  if (data.error) {
    detailTr.querySelector('.detail-wrap').innerHTML = '<span style="color:#f87171">' + esc(data.error) + '</span>';
    return;
  }
  var wrap = detailTr.querySelector('.detail-wrap');
  var statRow = document.createElement('div');
  statRow.className = 'stat-row';
  statRow.innerHTML = statBox(fmt(data.price), 'Price') + statBox(data.appearances, 'Appearances') +
    statBox(data.picked, 'Times Picked') + statBox(data.skipped, 'Skipped', '#6b7280');
  wrap.innerHTML = '';
  wrap.appendChild(statRow);
  var innerTbody = document.createElement('tbody');
  if (!data.rows.length) {
    innerTbody.innerHTML = '<tr><td colspan="5" style="color:#8b949e;padding:.3rem .6rem">Nobody has picked this item.</td></tr>';
  } else {
    data.rows.forEach(function(r) {
      var row = document.createElement('tr');
      row.innerHTML = '<td>' + esc(r.player) + '</td>' +
        '<td class="td-right">' + r.picks + '</td>' +
        '<td class="td-right">' + r.wins + '</td>' +
        '<td class="td-right">' + r.collisions + '</td>' +
        '<td class="td-right">' + pct(r.pick_rate) + '</td>';
      innerTbody.appendChild(row);
    });
  }
  var innerTable = document.createElement('table');
  innerTable.innerHTML = '<thead><tr><th>Player</th><th class="td-right">Picks</th><th class="td-right">Wins</th><th class="td-right">Collisions</th><th class="td-right">Pick Rate</th></tr></thead>';
  innerTable.appendChild(innerTbody);
  wrap.appendChild(innerTable);
}

panelLoaders['items-list'] = async function() {
  _itemsData = await (await fetch('/api/stats/items')).json();
  _itemsSort = {col: 'price', dir: -1};
  _itemFilter = '';
  var searchEl = document.getElementById('items-search');
  if (searchEl) searchEl.value = '';
  renderItems();
};

async function renameItem(inp) {
  var newName = inp.value.trim();
  var oldName = inp.dataset.orig;
  if (!newName || newName === oldName) { inp.value = oldName; return; }
  var data = await post('/api/items/rename', {old_name: oldName, new_name: newName});
  if (data.error) { showMsg('items-list-msg', esc(data.error), 'err'); inp.value = oldName; return; }
  clearMsg('items-list-msg');
  panelLoaders['items-list']();
}

async function updateItemPrice(inp) {
  var raw   = inp.value.replace(/,/g, '').trim();
  var price = parseFloat(raw);
  if (isNaN(price) || price < 0) { inp.value = fmt(parseFloat(inp.dataset.orig)); return; }
  if (price === parseFloat(inp.dataset.orig)) { inp.value = fmt(price); return; }
  var data = await post('/api/items/update_price', {item: inp.dataset.item, price: price});
  if (data.error) { showMsg('items-list-msg', esc(data.error), 'err'); inp.value = fmt(parseFloat(inp.dataset.orig)); return; }
  clearMsg('items-list-msg');
  inp.dataset.orig = price;
  inp.value = fmt(price);
}


// ---------------------------------------------------------------------------
// My Stats
// ---------------------------------------------------------------------------
panelLoaders['mystats'] = async function() {
  var data = await (await fetch('/api/mystats')).json();
  var content = document.getElementById('mystats-content');
  if (data.error) {
    showMsg('mystats-msg', esc(data.error) +
      ' Go to Settings to set your player name.', 'err');
    content.classList.add('hidden'); return;
  }
  clearMsg('mystats-msg');
  document.getElementById('mystats-stat-row').innerHTML =
    statBox(data.rounds_played, 'Rounds') +
    statBox(data.picks, 'Picks') +
    statBox(data.wins, 'Wins', '#4ade80') +
    statBox(data.collisions, 'Collisions', '#f87171') +
    statBox(pct(data.win_rate), 'Win Rate');
  personalRows(document.getElementById('mystats-tbody'), data.rows);
  content.classList.remove('hidden');
};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
panelLoaders['settings'] = async function() {
  var data = await (await fetch('/api/settings')).json();
  document.getElementById('inp-my-player').value = data.my_player || '';
  document.getElementById('sel-utility').value   = data.utility_mode || 'linear';
  document.getElementById('inp-decay').value     = data.decay_factor || 0.8;

};

async function savePlayerName() {
  var name = document.getElementById('inp-my-player').value.trim();
  if (!name) return;
  var data = await post('/api/settings/player', {name: name});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', "Player name set to '" + esc(data.name) + "'.");
  var info = document.getElementById('topbar-info');
  var rounds = info.textContent.split('·')[0].trim();
  info.textContent = rounds + ' · ' + data.name;
}

async function saveUtility() {
  var utility = document.getElementById('sel-utility').value;
  var data = await post('/api/settings/utility', {utility: utility});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', "Utility mode set to '" + data.utility + "'.");
}

async function saveDecay() {
  var decay = parseFloat(document.getElementById('inp-decay').value);
  var data = await post('/api/settings/decay', {decay: decay});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', 'Decay factor set to ' + data.decay + '. Model retrained.');
}

async function mergePlayers() {
  var from_name = document.getElementById('inp-merge-from').value.trim();
  var to_name   = document.getElementById('inp-merge-into').value.trim();
  if (!from_name || !to_name) {
    showMsg('settings-msg', 'Both names required.', 'err'); return;
  }
  if (!confirm("Merge '" + from_name + "' → '" + to_name + "'?\\nThis rewrites all historical round data and records an alias so the old name is recognised automatically in future.")) return;
  var data = await post('/api/settings/merge', {from_name: from_name, to_name: to_name});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', "Merged '" + esc(data.from) + "' → '" + esc(data.to) +
    "'. " + data.changed + " occurrence(s) updated.");
  document.getElementById('inp-merge-from').value = '';
  document.getElementById('inp-merge-into').value = '';
}

async function mergeItems() {
  var from_name = document.getElementById('inp-item-merge-from').value.trim();
  var to_name   = document.getElementById('inp-item-merge-into').value.trim();
  if (!from_name || !to_name) {
    showMsg('settings-msg', 'Both fields required.', 'err'); return;
  }
  if (!confirm("Merge '" + from_name + "' → '" + to_name + "'?\\nThis rewrites all historical round data and removes the first item.")) return;
  var data = await post('/api/items/merge', {from_name: from_name, to_name: to_name});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', "Merged '" + esc(data.from) + "' → '" + esc(data.to) +
    "'. " + data.changed + " round(s) updated.");
  document.getElementById('inp-item-merge-from').value = '';
  document.getElementById('inp-item-merge-into').value = '';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("  Nobody Wants It — Web UI")
    print("  Open http://localhost:5001 in your browser.")
    print("  Press Ctrl-C to stop.")
    app.run(host="127.0.0.1", port=5001, debug=False)
