##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## tiebreaker_cli
##

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

def parse_rank_date_col(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r'\.0$', '', regex=True)
    dt1 = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    dt2 = pd.to_datetime(s, errors="coerce")
    return dt1.fillna(dt2).dt.date

class DataHub:
    def __init__(self, data_root: Path):
        self.root = data_root
        self.players = None

    def load_players(self) -> pd.DataFrame:
        if self.players is not None:
            return self.players
        p = self.root / "atp_player" / "atp_players.csv"
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")
        df = pd.read_csv(p, low_memory=False)
        cols = {c.lower(): c for c in df.columns}
        first = cols.get("name_first") or cols.get("firstname") or cols.get("first_name")
        last = cols.get("name_last") or cols.get("lastname") or cols.get("last_name")
        player = cols.get("player") or cols.get("name")
        if first and last:
            df["full_name"] = (df[first].fillna('') + " " + df[last].fillna('')).str.strip()
        elif player:
            df["full_name"] = df[player].astype(str)
        else:
            raise ValueError("Impossible d'inférer la colonne du nom dans atp_players.csv")
        pid_col = cols.get("player_id") or cols.get("id") or "player_id"
        if pid_col not in df.columns:
            raise ValueError("Colonne player_id introuvable dans atp_players.csv")
        df = df.rename(columns={pid_col: "player_id"})
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        self.players = df
        return df

    def load_rankings(self) -> pd.DataFrame:
        parts = []
        cur = self.root / "atp_current_ranking" / "atp_rankings_current.csv"
        old = self.root / "atp_old_ranking"
        if cur.exists():
            parts.append(pd.read_csv(cur, low_memory=False))
        if old.exists():
            for f in sorted(old.glob("atp_rankings_*s.csv")):
                parts.append(pd.read_csv(f, low_memory=False))
        if not parts:
            raise FileNotFoundError("Aucun fichier de ranking trouvé sous data/atp_current_ranking ou data/atp_old_ranking")

        df = pd.concat(parts, ignore_index=True)
        cols = {c.lower(): c for c in df.columns}
        rd = cols.get("ranking_date") or "ranking_date"
        if rd in df.columns:
            df["ranking_date"] = parse_rank_date_col(df[rd])

        rename = {}
        if "player" in cols:
            player_col = cols["player"]
            if pd.api.types.is_numeric_dtype(df[player_col]) or df[player_col].astype(str).str.isdigit().all():
                rename[player_col] = "player_id"
            else:
                rename[player_col] = "player_name_raw"
        if "player_id" in cols:
            rename[cols["player_id"]] = "player_id"
        if "rank" in cols:
            rename[cols["rank"]] = "rank"
        if "points" in cols:
            rename[cols["points"]] = "points"
        if rd in df.columns:
            rename[rd] = "ranking_date"
        df = df.rename(columns=rename)

        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        if "rank" in df.columns:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
        if "points" in df.columns:
            df["points"] = pd.to_numeric(df["points"], errors="coerce").astype("Int64")
        return df

    def load_matches(self, years: list[int] | None = None) -> pd.DataFrame:
        matches_dir = self.root / "atp_matches"
        files = []
        if years:
            for y in years:
                f = matches_dir / f"atp_matches_{y}.csv"
                if f.exists():
                    files.append(f)
        else:
            files = [p for p in matches_dir.glob("atp_matches_*.csv") if re.search(r"\d{4}\.csv$", p.name)]
        if not files:
            raise FileNotFoundError("Aucun fichier de matches singles trouvé (atp_matches_YYYY.csv).")

        dfs = [pd.read_csv(f, low_memory=False) for f in sorted(files)]
        df = pd.concat(dfs, ignore_index=True)
        cols = {c.lower(): c for c in df.columns}
        tdate = cols.get("tourney_date") or "tourney_date"
        if tdate in df.columns:
            def parse_date(v):
                try:
                    s = str(int(v))
                    return datetime.strptime(s, "%Y%m%d").date()
                except Exception:
                    try:
                        return pd.to_datetime(v, errors="coerce").date()
                    except Exception:
                        return pd.NaT
            df["tourney_date"] = df[tdate].apply(parse_date)
        for k in ["winner_name", "loser_name", "tourney_name", "round", "score", "surface", "minutes", "best_of"]:
            if k not in df.columns and k in cols:
                df = df.rename(columns={cols[k]: k})
        for k in ["winner_name", "loser_name", "tourney_name", "round", "score", "surface"]:
            if k in df.columns:
                df[k] = df[k].astype(str)
        return df

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
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Dossier racine des données (défaut: ./data)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_rank = sp.add_parser("rank", help="Obtenir le rang ATP d'un joueur à une date donnée (ou le plus récent)")
    ap_rank.add_argument("--player", required=True, help="Nom du joueur (ex: 'Novak Djokovic')")
    ap_rank.add_argument("--date", help="Date ISO (YYYY-MM-DD). Si absente, prend le dernier ranking disponible (current sinon historique).")
    ap_rank.set_defaults(func=cmd_rank)

    ap_match = sp.add_parser("match", help="Trouver le résultat d'un match précis entre deux joueurs")
    ap_match.add_argument("--p1", required=True, help="Joueur 1 (ordre indifférent)")
    ap_match.add_argument("--p2", required=True, help="Joueur 2 (ordre indifférent)")
    ap_match.add_argument("--year", help="Année exacte (ex: 2023). Accélère la recherche.")
    ap_match.add_argument("--tournament", help="Filtre sur le nom du tournoi (contient)")
    ap_match.add_argument("--round", help="Filtre round exact (ex: F, SF, QF, R16, R32, R64, R128)")
    ap_match.add_argument("--surface", help="Filtre surface exacte (Hard, Clay, Grass, Carpet)")
    ap_match.add_argument("--date", help="Filtre date exacte du match/tournoi (YYYY-MM-DD)")
    ap_match.add_argument("--all-years", action="store_true", help="Parcourir toutes les années (lent) si --year absent")
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
