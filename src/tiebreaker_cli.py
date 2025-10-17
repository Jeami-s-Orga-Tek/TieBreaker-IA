##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## tiebreaker_cli
##

from .models import DataHub
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from difflib import get_close_matches

def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().casefold())

def best_name_match(query: str, candidates: list[str]) -> str | None:
    q = norm(query)
    for c in candidates:
        if norm(c) == q:
            return c
    m = get_close_matches(query, candidates, n=1, cutoff=0.75)
    return m[0] if m else None

def date_parse_or_none(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None

def resolve_player_id(hub: DataHub, name_query: str):
    players = hub.load_players()
    candidates = players["full_name"].astype(str).tolist()
    match = best_name_match(name_query, candidates)
    if not match:
        return None, None
    row = players[players["full_name"] == match].iloc[0]
    return (int(row["player_id"]) if pd.notna(row["player_id"]) else None), str(row["full_name"])

def cmd_rank(args, hub: DataHub):
    pid, resolved = resolve_player_id(hub, args.player)
    if pid is None:
        print(f"Joueur introuvable: {args.player}", file=sys.stderr)
        return 1
    rankings = hub.load_rankings()

    if "player_id" in rankings.columns:
        df = rankings[rankings["player_id"] == pid]
    else:
        if "player_name_raw" in rankings.columns:
            firstname = resolved.split(" ")[0]
            lastname = resolved.split(" ")[-1]
            variants = {f"{lastname}, {firstname}".strip(), f"{firstname} {lastname}".strip(), resolved}
            df = rankings[rankings["player_name_raw"].astype(str).isin(variants)]
        else:
            df = pd.DataFrame()

    if df.empty:
        print(f"Aucun ranking trouvé pour {resolved} (player_id={pid}).")
        return 0

    target_date = date_parse_or_none(args.date)
    if "ranking_date" in df.columns and df["ranking_date"].notna().any():
        df = df.dropna(subset=["ranking_date"]).sort_values("ranking_date")
        if target_date:
            df = df[df["ranking_date"] <= target_date]
            if df.empty:
                print(f"Aucun ranking pour {resolved} avant {args.date}.")
                return 0
        row = df.iloc[-1]
        date_str = row["ranking_date"].isoformat()
    else:
        row = df.iloc[-1]
        date_str = "(date inconnue)"

    rank = int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else None
    points = int(row["points"]) if "points" in row and pd.notna(row["points"]) else None

    if rank is not None and points is not None:
        print(f"{resolved} — Rang ATP {rank} ({points} pts) au {date_str}")
    elif rank is not None:
        print(f"{resolved} — Rang ATP {rank} au {date_str}")
    else:
        print(f"Ranking introuvable pour {resolved} (au {date_str}).")
    return 0

def cmd_match(args, hub: DataHub):
    pid1, p1 = resolve_player_id(hub, args.p1)
    pid2, p2 = resolve_player_id(hub, args.p2)
    if pid1 is None or pid2 is None:
        if pid1 is None:
            print(f"Joueur P1 introuvable: {args.p1}", file=sys.stderr)
        if pid2 is None:
            print(f"Joueur P2 introuvable: {args.p2}", file=sys.stderr)
        return 1

    years = None
    if args.year:
        try:
            years = [int(args.year)]
        except Exception:
            print("--year doit être un entier (ex: 2023)", file=sys.stderr)
            return 1
    elif not args.all_years:
        this_year = datetime.utcnow().year
        years = list(range(this_year - 9, this_year + 1))

    matches = hub.load_matches(years=years)

    def name_match_col(col: str, target: str) -> pd.Series:
        return matches[col].str.casefold().str.strip() == target.casefold().strip()

    mask_pair = ( (name_match_col("winner_name", p1) & name_match_col("loser_name", p2)) | (name_match_col("winner_name", p2) & name_match_col("loser_name", p1)) )
    df = matches[mask_pair].copy()
    if args.tournament:
        df = df[df["tourney_name"].str.contains(args.tournament, case=False, na=False)]
    if args.round:
        df = df[df["round"].str.fullmatch(args.round, case=False, na=False)]
    if args.surface:
        df = df[df["surface"].str.fullmatch(args.surface, case=False, na=False)]
    if args.date:
        d = date_parse_or_none(args.date)
        if d:
            df = df[df["tourney_date"] == d]

    if df.empty:
        scope = f" (années {min(years)}-{max(years)})" if years else ""
        print(f"Aucun match {p1} vs {p2}{scope} avec ces filtres.")
        return 0

    if "tourney_date" in df.columns:
        df = df.sort_values(["tourney_date", "tourney_name", "round"], na_position="last")
    else:
        df = df.sort_values(["tourney_name", "round"], na_position="last")

    cols = df.columns
    have_minutes = "minutes" in cols
    have_score = "score" in cols
    have_best_of = "best_of" in cols
    have_round = "round" in cols
    have_surface = "surface" in cols

    def row_to_str(r):
        date_str = r["tourney_date"].isoformat() if pd.notna(r.get("tourney_date")) else "????-??-??"
        parts = [f"{date_str} — {r.get('tourney_name','?')}"]
        if have_surface and pd.notna(r.get("surface")):
            parts[-1] += f" ({r['surface']})"
        if have_round and pd.notna(r.get("round")):
            parts.append(f"R: {r['round']}")
        if have_best_of and pd.notna(r.get("best_of")):
            try:
                parts.append(f"Best-of-{int(r['best_of'])}")
            except Exception:
                pass
        wl = f"{r.get('winner_name','?')} def. {r.get('loser_name','?')}"
        if have_score and pd.notna(r.get('score')):
            wl += f"  {r['score']}"
        if have_minutes and pd.notna(r.get('minutes')):
            try:
                wl += f"  ({int(r['minutes'])} min)"
            except Exception:
                pass
        return " | ".join(parts) + " | " + wl

    for _, r in df.iterrows():
        print(row_to_str(r))
    return 0

def build_parser():
    ap = argparse.ArgumentParser(description="TieBreaker CLI — Parser ATP (rankings & matches)")
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Data root directory (default: ./data)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_rank = sp.add_parser("rank", help="Get a player's ATP ranking on a given date (or the most recent one)")
    ap_rank.add_argument("--player", required=True, help="Player name (ex: 'Novak Djokovic')")
    ap_rank.add_argument("--date", help="Date ISO (YYYY-MM-DD). If absent, take the last available ranking (current if available, otherwise historical).")
    ap_rank.set_defaults(func=cmd_rank)

    ap_match = sp.add_parser("match", help="Find the result of a specific match between two players")
    ap_match.add_argument("--p1", required=True, help="Player 1 (indifferent order)")
    ap_match.add_argument("--p2", required=True, help="Player 2 (indifferent order)")
    ap_match.add_argument("--year", help="Exact year (ex: 2023). Speed up your search.")
    ap_match.add_argument("--tournament", help="Filter by tournament name (contains)")
    ap_match.add_argument("--round", help="Exact round filter (ex: F, SF, QF, R16, R32, R64, R128)")
    ap_match.add_argument("--surface", help="Exact surface filter (Hard, Clay, Grass, Carpet)")
    ap_match.add_argument("--date", help="Exact date filter for match/tournament (YYYY-MM-DD)")
    ap_match.add_argument("--all-years", action="store_true", help="Browse all years (slow) if --year is absent")
    ap_match.set_defaults(func=cmd_match)
    return ap

def main(argv=None):
    argv = argv or sys.argv[1:]
    ap = build_parser()
    args = ap.parse_args(argv)
    hub = DataHub(args.data_root)
    return args.func(args, hub)

if __name__ == "__main__":
    raise SystemExit(main())
