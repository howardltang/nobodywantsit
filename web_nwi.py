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

    _state["rounds"].append(new_round)
    save_state(_state)
    _retrain()
    _pending.clear()

    return jsonify({"ok": True, "winners": winners, "round_num": len(_state["rounds"])})


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

    save_state(_state)
    _retrain()
    return jsonify({"ok": True, "changed": changed, "from": from_canon, "to": to_canon})


@app.route("/api/settings/alias/add", methods=["POST"])
def api_settings_alias_add():
    data      = request.get_json()
    alias     = data.get("alias",     "").strip()
    canonical = data.get("canonical", "").strip()
    if not alias or not canonical:
        return jsonify({"error": "Both fields required."}), 400
    if alias.lower() == canonical.lower():
        return jsonify({"error": "Alias and canonical name are the same."}), 400
    _state.setdefault("name_aliases", {})[alias.lower()] = canonical
    save_state(_state)
    return jsonify({"ok": True})


@app.route("/api/settings/alias/remove", methods=["POST"])
def api_settings_alias_remove():
    alias   = request.get_json().get("alias", "").strip().lower()
    aliases = _state.get("name_aliases", {})
    if alias not in aliases:
        return jsonify({"error": f"Alias '{alias}' not found."}), 400
    del aliases[alias]
    save_state(_state)
    return jsonify({"ok": True})


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

    /* Settings */
    .settings-sep { font-size: .75rem; color: #484f58; text-transform: uppercase;
                    letter-spacing: .05em; border-bottom: 1px solid #30363d;
                    padding-bottom: .3rem; margin: 1rem 0 .6rem; }
    .alias-row { display: flex; align-items: center; gap: .4rem; padding: .2rem .4rem;
                 background: #0d1117; border-radius: 3px; margin-bottom: .3rem; font-size: .82rem; }
    .alias-arrow { color: #58a6ff; }
    .alias-name  { flex: 1; }
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
    <button class="nav-btn" onclick="showPanel('items-list', this)">Item Prices</button>
    <button class="nav-btn" onclick="showPanel('browse-players', this)">Browse Players</button>
    <button class="nav-btn" onclick="showPanel('browse-items', this)">Browse Items</button>
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
        <h6>Play a Round</h6>
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
        <h6>Round Saved!</h6>
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
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th>Player</th>
              <th class="td-right">Picks</th><th class="td-right">Wins</th>
              <th class="td-right">Win%</th><th class="td-right">Contrarian</th>
            </tr></thead>
            <tbody id="players-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ─── ITEM PRICES ─── -->
    <div class="panel" id="panel-items-list">
      <div class="card">
        <h6>Item Prices</h6>
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th>Item</th>
              <th class="td-right">Price</th><th class="td-right">App.</th>
              <th class="td-right">Picked</th><th class="td-right">Solo</th>
              <th class="td-right">Avg Pickers</th>
            </tr></thead>
            <tbody id="items-list-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ─── BROWSE PLAYERS ─── -->
    <div class="panel" id="panel-browse-players">
      <div class="card">
        <h6>Browse Player Stats</h6>
        <div class="search-row">
          <input type="text" id="player-search-inp" placeholder="Search player name..."
            oninput="debouncedSearch('player', this.value)"
            onkeydown="if(event.key==='Enter') lookupPlayer()">
          <button class="btn btn-primary" onclick="lookupPlayer()">Look Up</button>
        </div>
        <div class="suggestions hidden" id="player-suggestions"></div>
        <div id="player-detail-msg"></div>
        <div id="player-detail" class="hidden">
          <div class="stat-row" id="player-stat-row"></div>
          <div class="tbl-wrap">
            <table>
              <thead><tr>
                <th>Rd</th><th>Item</th><th>Outcome</th>
                <th class="td-right">Others</th><th class="td-right">Value</th>
              </tr></thead>
              <tbody id="player-detail-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- ─── BROWSE ITEMS ─── -->
    <div class="panel" id="panel-browse-items">
      <div class="card">
        <h6>Browse Item History</h6>
        <div class="search-row">
          <input type="text" id="item-search-inp" placeholder="Search item name..."
            oninput="debouncedSearch('item', this.value)"
            onkeydown="if(event.key==='Enter') lookupItem()">
          <button class="btn btn-primary" onclick="lookupItem()">Look Up</button>
        </div>
        <div class="suggestions hidden" id="item-suggestions"></div>
        <div id="item-detail-msg"></div>
        <div id="item-detail" class="hidden">
          <div class="stat-row" id="item-stat-row"></div>
          <div class="tbl-wrap">
            <table>
              <thead><tr>
                <th>Player</th>
                <th class="td-right">Picks</th><th class="td-right">Wins</th>
                <th class="td-right">Collisions</th><th class="td-right">Pick Rate</th>
              </tr></thead>
              <tbody id="item-detail-tbody"></tbody>
            </table>
          </div>
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

        <div class="settings-sep">Name Aliases</div>
        <div id="alias-list" style="margin-bottom:.6rem"></div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:flex-end">
          <div class="field" style="flex:1;min-width:100px;margin-bottom:0">
            <label>Alias (auto-replaced on input)</label>
            <input type="text" id="inp-alias" placeholder="Neuv">
          </div>
          <div style="color:#8b949e;padding-bottom:.4rem">→</div>
          <div class="field" style="flex:1;min-width:100px;margin-bottom:0">
            <label>Canonical name</label>
            <input type="text" id="inp-alias-canon" placeholder="Neuvillette">
          </div>
        </div>
        <button class="btn btn-primary btn-sm" style="margin-top:.5rem" onclick="addAlias()">Add Alias</button>
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
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
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
      '<td class="td-right td-mono">' + fmt(r.value) + '</td>' +
      '<td class="td-right td-mono ' + psClass + '">' + pct(r.p_solo) + '</td>' +
      '<td class="td-right td-mono">' + evStr + '</td>' +
      '<td style="color:#8b949e;font-size:.75rem">' + esc(r.history) + '</td>';
    tbody.appendChild(tr);
  });

  document.getElementById('rec-footer').textContent =
    '★ = top pick  |  ' + data.utility_desc + '  |  YOU: W=won, C=collision';
  document.getElementById('rec-output').classList.remove('hidden');
}

async function saveRound() {
  var results = [];
  document.querySelectorAll('.result-row').forEach(function(row) {
    var val = row.querySelector('.result-players').value.trim();
    if (val) {
      var players = val.split(',').map(function(p){ return p.trim(); }).filter(Boolean);
      if (players.length) results.push({item: row.dataset.item, players: players});
    }
  });

  var data = await post('/api/round/save', {results: results});
  if (data.error) { showMsg('round-msg', esc(data.error), 'err'); return; }

  var display = document.getElementById('winners-display');
  if (!data.winners.length) {
    display.innerHTML = '<div class="card-sub">No solo winners this round.</div>';
  } else {
    display.innerHTML = data.winners.map(function(w) {
      return '<div class="winner-row">🏆 <strong>' + esc(w.player) +
        '</strong> won <strong>' + esc(w.item) + '</strong>' +
        '<span class="winner-val">' + fmt(w.value) + ' meat</span></div>';
    }).join('');
  }

  // Update topbar
  var info = document.getElementById('topbar-info');
  var mp = info.textContent.indexOf('·');
  var suffix = mp !== -1 ? ' ' + info.textContent.substring(mp) : '';
  info.textContent = data.round_num + ' round' + (data.round_num !== 1 ? 's' : '') + suffix;

  document.getElementById('step-recommend').classList.add('hidden');
  document.getElementById('step-results').classList.add('hidden');
  document.getElementById('step-winners').classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Player Leaderboard
// ---------------------------------------------------------------------------
panelLoaders['players'] = async function() {
  var rows = await (await fetch('/api/stats/players')).json();
  var tbody = document.getElementById('players-tbody');
  tbody.innerHTML = '';
  rows.forEach(function(r) {
    var tr = document.createElement('tr');
    tr.innerHTML = '<td>' + esc(r.name) + '</td>' +
      '<td class="td-right">' + r.picks + '</td>' +
      '<td class="td-right">' + r.wins  + '</td>' +
      '<td class="td-right">' + pct(r.win_rate) + '</td>' +
      '<td class="td-right">' + r.contrarian.toFixed(2) + '</td>';
    tbody.appendChild(tr);
  });
};

// ---------------------------------------------------------------------------
// Item Prices
// ---------------------------------------------------------------------------
panelLoaders['items-list'] = async function() {
  var rows = await (await fetch('/api/stats/items')).json();
  var tbody = document.getElementById('items-list-tbody');
  tbody.innerHTML = '';
  rows.forEach(function(r) {
    var tr = document.createElement('tr');
    tr.innerHTML = '<td>' + esc(r.item) + '</td>' +
      '<td class="td-right td-mono">' + fmt(r.price) + '</td>' +
      '<td class="td-right">' + r.appearances + '</td>' +
      '<td class="td-right">' + r.picked + '</td>' +
      '<td class="td-right">' + r.solo + '</td>' +
      '<td class="td-right">' + (r.avg_pickers > 0 ? r.avg_pickers.toFixed(1) : '—') + '</td>';
    tbody.appendChild(tr);
  });
};

// ---------------------------------------------------------------------------
// Search debounce (shared)
// ---------------------------------------------------------------------------
var _searchTimers = {};
function debouncedSearch(type, val) {
  clearTimeout(_searchTimers[type]);
  var sugEl = document.getElementById(type + '-suggestions');
  if (!val) { sugEl.classList.add('hidden'); return; }
  _searchTimers[type] = setTimeout(async function() {
    var url = '/api/stats/search/' + (type === 'player' ? 'players' : 'items') +
              '?q=' + encodeURIComponent(val);
    var names = await (await fetch(url)).json();
    if (!names.length) { sugEl.classList.add('hidden'); return; }
    sugEl.innerHTML = names.map(function(n) {
      return '<button class="btn btn-secondary btn-sm" style="margin:.1rem" ' +
        'data-name="' + esc(n) + '" onclick="selectSuggestion(\\'' + type + '\\', this.dataset.name)">' +
        esc(n) + '</button>';
    }).join('');
    sugEl.classList.remove('hidden');
  }, 250);
}

function selectSuggestion(type, name) {
  document.getElementById(type + '-search-inp').value = name;
  document.getElementById(type + '-suggestions').classList.add('hidden');
  if (type === 'player') lookupPlayer();
  else lookupItem();
}

// ---------------------------------------------------------------------------
// Browse Players
// ---------------------------------------------------------------------------
async function lookupPlayer() {
  var name = document.getElementById('player-search-inp').value.trim();
  if (!name) return;
  var data = await (await fetch('/api/stats/player/' + encodeURIComponent(name))).json();
  var detailEl = document.getElementById('player-detail');
  if (data.error) {
    showMsg('player-detail-msg', esc(data.error), 'err');
    detailEl.classList.add('hidden'); return;
  }
  clearMsg('player-detail-msg');
  document.getElementById('player-stat-row').innerHTML =
    statBox(data.rounds_played, 'Rounds') +
    statBox(data.picks, 'Picks') +
    statBox(data.wins, 'Wins', '#4ade80') +
    statBox(data.collisions, 'Collisions', '#f87171') +
    statBox(pct(data.win_rate), 'Win Rate');
  personalRows(document.getElementById('player-detail-tbody'), data.rows);
  detailEl.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Browse Items
// ---------------------------------------------------------------------------
async function lookupItem() {
  var name = document.getElementById('item-search-inp').value.trim();
  if (!name) return;
  var data = await (await fetch('/api/stats/item/' + encodeURIComponent(name))).json();
  var detailEl = document.getElementById('item-detail');
  if (data.error) {
    showMsg('item-detail-msg', esc(data.error), 'err');
    detailEl.classList.add('hidden'); return;
  }
  clearMsg('item-detail-msg');
  document.getElementById('item-stat-row').innerHTML =
    statBox(fmt(data.price), 'Price') +
    statBox(data.appearances, 'Appearances') +
    statBox(data.picked, 'Times Picked') +
    statBox(data.skipped, 'Skipped', '#6b7280');

  var tbody = document.getElementById('item-detail-tbody');
  tbody.innerHTML = '';
  if (!data.rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="color:#8b949e;padding:.5rem .6rem">Nobody has picked this item.</td></tr>';
  } else {
    data.rows.forEach(function(r) {
      var tr = document.createElement('tr');
      tr.innerHTML = '<td>' + esc(r.player) + '</td>' +
        '<td class="td-right">' + r.picks + '</td>' +
        '<td class="td-right">' + r.wins + '</td>' +
        '<td class="td-right">' + r.collisions + '</td>' +
        '<td class="td-right">' + pct(r.pick_rate) + '</td>';
      tbody.appendChild(tr);
    });
  }
  detailEl.classList.remove('hidden');
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

  var aliases = data.aliases || {};
  var aliasList = document.getElementById('alias-list');
  var keys = Object.keys(aliases);
  if (!keys.length) {
    aliasList.innerHTML = '<div style="color:#8b949e;font-size:.8rem">(no aliases defined)</div>';
  } else {
    aliasList.innerHTML = keys.map(function(alias) {
      return '<div class="alias-row">' +
        '<span class="alias-name">&#39;' + esc(alias) + '&#39;</span>' +
        '<span class="alias-arrow">→</span>' +
        '<span class="alias-name">&#39;' + esc(aliases[alias]) + '&#39;</span>' +
        '<button class="btn btn-danger btn-sm" data-alias="' + esc(alias) +
        '" onclick="removeAlias(this.dataset.alias)">Remove</button></div>';
    }).join('');
  }
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
  if (!confirm("Merge '" + from_name + "' → '" + to_name + "'?\\nThis rewrites all historical round data.")) return;
  var data = await post('/api/settings/merge', {from_name: from_name, to_name: to_name});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', "Merged '" + esc(data.from) + "' → '" + esc(data.to) +
    "'. " + data.changed + " occurrence(s) updated.");
  document.getElementById('inp-merge-from').value = '';
  document.getElementById('inp-merge-into').value = '';
}

async function addAlias() {
  var alias     = document.getElementById('inp-alias').value.trim();
  var canonical = document.getElementById('inp-alias-canon').value.trim();
  if (!alias || !canonical) {
    showMsg('settings-msg', 'Both fields required.', 'err'); return;
  }
  var data = await post('/api/settings/alias/add', {alias: alias, canonical: canonical});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', "Alias '" + esc(alias) + "' → '" + esc(canonical) + "' added.");
  document.getElementById('inp-alias').value       = '';
  document.getElementById('inp-alias-canon').value = '';
  panelLoaders['settings']();
}

async function removeAlias(alias) {
  var data = await post('/api/settings/alias/remove', {alias: alias});
  if (data.error) { showMsg('settings-msg', esc(data.error), 'err'); return; }
  showMsg('settings-msg', "Alias '" + esc(alias) + "' removed.");
  panelLoaders['settings']();
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("  Nobody Wants It — Web UI")
    print("  Open http://localhost:5001 in your browser.")
    print("  Press Ctrl-C to stop.")
    app.run(host="127.0.0.1", port=5001, debug=False)
