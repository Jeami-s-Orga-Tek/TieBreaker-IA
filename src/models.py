##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## models
##

import re
from pathlib import Path
from datetime import datetime
import pandas as pd

from .parser import parse_rank_date_col

# class DecisionTreeModel:

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
            files = [p for p in matches_dir.glob("atp_matches_*.csv") if re.fullmatch(r"atp_matches_\d{4}\.csv", p.name)]
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
