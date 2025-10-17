"""
TieBreaker dataset builder — constructs an A vs B modelling table from ATP matches.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .models import DataHub
from .features_recent import add_recent_form_features


@dataclass(slots=True)
class PlayerLookup:
    dob_by_id: dict[int, pd.Timestamp]
    dob_by_name: dict[str, pd.Timestamp]
    name_by_id: dict[int, str]
    name_by_key: dict[str, str]


_RANKING_CACHE: dict[int, dict[str, Any]] = {}


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().split()).casefold()


def parse_date_like(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, date):
        return pd.Timestamp(value)
    return pd.to_datetime(value, errors="coerce")


def parse_dob_value(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        text = f"{int(value):08d}"
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return pd.NaT
        text = f"{int(value):08d}"
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    text = str(value).strip()
    if not text:
        return pd.NaT
    cleaned = text.replace("-", "").replace("/", "").replace(".", "")
    if len(cleaned) == 8 and cleaned.isdigit():
        return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def prepare_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    df = rankings.copy()
    if "ranking_date" in df.columns:
        df["ranking_date"] = pd.to_datetime(df["ranking_date"], errors="coerce")
        df = df.dropna(subset=["ranking_date"])
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    if "points" in df.columns:
        df["points"] = pd.to_numeric(df["points"], errors="coerce").astype("Int64")
    if "player_name_raw" in df.columns:
        df["__name_key"] = df["player_name_raw"].astype(str).map(normalize_name)
    return df


def _build_ranking_cache(df: pd.DataFrame) -> dict[str, Any]:
    by_pid: dict[int, pd.DataFrame] = {}
    by_name: dict[str, pd.DataFrame] = {}
    if "player_id" in df.columns:
        grouped = df.sort_values(["player_id", "ranking_date"]).groupby("player_id")
        for pid, group in grouped:
            if pd.isna(pid):
                continue
            by_pid[int(pid)] = group[["ranking_date", "rank", "points"]].reset_index(drop=True)
    if "__name_key" in df.columns:
        grouped = df.sort_values(["__name_key", "ranking_date"]).groupby("__name_key")
        for key, group in grouped:
            if not key:
                continue
            by_name[key] = group[["ranking_date", "rank", "points"]].reset_index(drop=True)
    return {"by_pid": by_pid, "by_name": by_name}


def _get_ranking_cache(df: pd.DataFrame) -> dict[str, Any]:
    cache = _RANKING_CACHE.get(id(df))
    if cache is None:
        cache = _build_ranking_cache(df)
        _RANKING_CACHE[id(df)] = cache
    return cache


def _select_latest_before(group: pd.DataFrame, target: pd.Timestamp, window_days: int | None = None) -> pd.Series | None:
    if group.empty:
        return None
    if pd.isna(target):
        return None if window_days is not None else group.iloc[-1]
    idx = group["ranking_date"].searchsorted(target, side="right") - 1
    if idx < 0:
        return None
    row = group.iloc[idx]
    if window_days is None:
        return row
    lower_bound = target - pd.Timedelta(days=window_days)
    ranking_date = row.get("ranking_date")
    if pd.isna(ranking_date) or ranking_date < lower_bound:
        return None
    return row


def _name_variants(player_name: str) -> list[str]:
    base = player_name or ""
    tokens = [t for t in base.strip().split() if t]
    variants = [base]
    if len(tokens) >= 2:
        first = tokens[0]
        last = tokens[-1]
        variants.append(f"{last}, {first}")
        variants.append(f"{last} {first}")
        variants.append(f"{last.upper()}, {first}")
        variants.append(f"{first} {last}")
    return list(dict.fromkeys(v for v in variants if v))


def get_rank_on_or_before(rankings: pd.DataFrame, player_id: int | None, player_name: str, match_date: date | datetime | pd.Timestamp | None) -> dict[str, Any]:
    """
    Retourne {'rank': int|NaN, 'points': int|NaN, 'ranking_date': date|NaT}
    """
    target = parse_date_like(match_date)
    if pd.isna(target):
        return {
            "rank": np.nan,
            "points": np.nan,
            "ranking_date": pd.NaT,
            "rank_missing": True,
            "points_missing": True,
        }
    cache = _get_ranking_cache(rankings)
    group: pd.DataFrame | None = None

    if player_id is not None:
        group = cache["by_pid"].get(player_id)

    if group is None and player_name:
        name_key = normalize_name(player_name)
        group = cache["by_name"].get(name_key)
        if group is None:
            for variant in _name_variants(player_name):
                group = cache["by_name"].get(normalize_name(variant))
                if group is not None:
                    break

    if group is None:
        return {"rank": np.nan, "points": np.nan, "ranking_date": pd.NaT, "rank_missing": True, "points_missing": True}

    row = _select_latest_before(group, target, window_days=365)
    if row is None:
        return {"rank": np.nan, "points": np.nan, "ranking_date": pd.NaT, "rank_missing": True, "points_missing": True}

    rank_value = row.get("rank")
    points_value = row.get("points")
    ranking_date = row.get("ranking_date")
    rank_missing = not pd.notna(rank_value)
    points_missing = not pd.notna(points_value)
    return {
        "rank": int(rank_value) if pd.notna(rank_value) else np.nan,
        "points": int(points_value) if pd.notna(points_value) else np.nan,
        "ranking_date": ranking_date if pd.notna(ranking_date) else pd.NaT,
        "rank_missing": rank_missing,
        "points_missing": points_missing,
    }


def prepare_players(players: pd.DataFrame) -> tuple[pd.DataFrame, PlayerLookup]:
    df = players.copy()
    if "dob" in df.columns:
        df["dob_parsed"] = df["dob"].apply(parse_dob_value)
    else:
        df["dob_parsed"] = pd.NaT
    df["name_key"] = df["full_name"].astype(str).map(normalize_name)
    dob_by_id: dict[int, pd.Timestamp] = {}
    dob_by_name: dict[str, pd.Timestamp] = {}
    name_by_id: dict[int, str] = {}
    name_by_key: dict[str, str] = {}
    for _, row in df.iterrows():
        pid = row.get("player_id")
        full_name = row.get("full_name")
        name_key = row.get("name_key")
        dob_value = row.get("dob_parsed")
        if pd.notna(pid):
            pid_int = int(pid)
            name_by_id[pid_int] = str(full_name) if full_name else ""
            if pd.notna(dob_value):
                dob_by_id[pid_int] = pd.Timestamp(dob_value)
        if isinstance(name_key, str) and name_key:
            name_by_key[name_key] = str(full_name) if full_name else ""
            if pd.notna(dob_value):
                dob_by_name[name_key] = pd.Timestamp(dob_value)
    lookup = PlayerLookup(dob_by_id=dob_by_id, dob_by_name=dob_by_name, name_by_id=name_by_id, name_by_key=name_by_key)
    return df, lookup


def _resolve_player_name(player_name: str, player_id: int | None, lookup: PlayerLookup) -> str:
    if player_id is not None and player_id in lookup.name_by_id:
        return lookup.name_by_id[player_id]
    name_key = normalize_name(player_name)
    return lookup.name_by_key.get(name_key, player_name)


def _resolve_player_dob(player_name: str, player_id: int | None, lookup: PlayerLookup) -> pd.Timestamp | None:
    if player_id is not None and player_id in lookup.dob_by_id:
        return lookup.dob_by_id[player_id]
    name_key = normalize_name(player_name)
    return lookup.dob_by_name.get(name_key)


def compute_age(dob: pd.Timestamp | None, match_date: pd.Timestamp | None) -> float:
    if dob is None or pd.isna(dob) or match_date is None or pd.isna(match_date):
        return np.nan
    delta_days = (match_date - dob).days
    if delta_days < 0:
        return np.nan
    return round(delta_days / 365.25, 2)


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def canonicalize_ab(row: Mapping[str, Any], rankings: pd.DataFrame, players_lookup: PlayerLookup) -> dict[str, Any]:
    match_date = parse_date_like(row.get("tourney_date"))
    winner_id = _safe_int(row.get("winner_id"))
    loser_id = _safe_int(row.get("loser_id"))
    winner_name = str(row.get("winner_name", "")).strip()
    loser_name = str(row.get("loser_name", "")).strip()

    winner_rank = get_rank_on_or_before(rankings, winner_id, winner_name, match_date)
    loser_rank = get_rank_on_or_before(rankings, loser_id, loser_name, match_date)

    winner_rank_value = winner_rank.get("rank")
    loser_rank_value = loser_rank.get("rank")

    winner_order = winner_rank_value if pd.notna(winner_rank_value) else np.inf
    loser_order = loser_rank_value if pd.notna(loser_rank_value) else np.inf

    winner_display_name = _resolve_player_name(winner_name, winner_id, players_lookup)
    loser_display_name = _resolve_player_name(loser_name, loser_id, players_lookup)

    if winner_order < loser_order:
        a_side = "winner"
    elif loser_order < winner_order:
        a_side = "loser"
    else:
        if normalize_name(winner_display_name) <= normalize_name(loser_display_name):
            a_side = "winner"
        else:
            a_side = "loser"

    if a_side == "winner":
        a_name = winner_display_name or winner_name
        b_name = loser_display_name or loser_name
        a_id = winner_id
        b_id = loser_id
        a_rank = winner_rank
        b_rank = loser_rank
        a_dob = _resolve_player_dob(winner_name, winner_id, players_lookup)
        b_dob = _resolve_player_dob(loser_name, loser_id, players_lookup)
        y_value = 1
    else:
        a_name = loser_display_name or loser_name
        b_name = winner_display_name or winner_name
        a_id = loser_id
        b_id = winner_id
        a_rank = loser_rank
        b_rank = winner_rank
        a_dob = _resolve_player_dob(loser_name, loser_id, players_lookup)
        b_dob = _resolve_player_dob(winner_name, winner_id, players_lookup)
        y_value = 0

    rank_a_value = a_rank.get("rank", np.nan)
    rank_b_value = b_rank.get("rank", np.nan)
    points_a_value = a_rank.get("points", np.nan)
    points_b_value = b_rank.get("points", np.nan)

    age_a = compute_age(a_dob, match_date)
    age_b = compute_age(b_dob, match_date)

    round_raw = str(row.get("round", "")).strip().upper() or None
    surface_raw = str(row.get("surface", "")).strip().title() or None

    best_of_raw = _safe_int(row.get("best_of"))
    best_of_inferred = 0
    if best_of_raw is None:
        level = str(row.get("tourney_level", "")).strip().upper()
        if level == "G":
            best_of_raw = 5
        else:
            best_of_raw = 3
        best_of_inferred = 1

    match_date_for_output: Any
    if isinstance(match_date, pd.Timestamp):
        match_date_for_output = match_date.normalize()
    else:
        match_date_for_output = match_date

    rank_missing_a = bool(a_rank.get("rank_missing", pd.isna(rank_a_value)))
    rank_missing_b = bool(b_rank.get("rank_missing", pd.isna(rank_b_value)))
    points_missing_a = bool(a_rank.get("points_missing", pd.isna(points_a_value)))
    points_missing_b = bool(b_rank.get("points_missing", pd.isna(points_b_value)))
    age_missing_a = int(np.isnan(age_a))
    age_missing_b = int(np.isnan(age_b))

    rank_diff_value = rank_a_value - rank_b_value if pd.notna(rank_a_value) and pd.notna(rank_b_value) else np.nan
    points_diff_value = points_a_value - points_b_value if pd.notna(points_a_value) and pd.notna(points_b_value) else np.nan
    age_diff_value = age_a - age_b if not np.isnan(age_a) and not np.isnan(age_b) else np.nan

    return {
        "A_name": a_name,
        "B_name": b_name,
        "A_player_id": a_id,
        "B_player_id": b_id,
        "y": y_value,
        "tourney_date": match_date_for_output,
        "tourney_name": row.get("tourney_name"),
        "surface": surface_raw,
        "round": round_raw,
        "best_of": best_of_raw,
        "best_of_inferred": best_of_inferred,
        "rank_A": float(rank_a_value) if pd.notna(rank_a_value) else np.nan,
        "rank_B": float(rank_b_value) if pd.notna(rank_b_value) else np.nan,
        "rank_diff": float(rank_diff_value) if pd.notna(rank_diff_value) else np.nan,
        "rank_missing_A": int(rank_missing_a or pd.isna(rank_a_value)),
        "rank_missing_B": int(rank_missing_b or pd.isna(rank_b_value)),
        "points_A": float(points_a_value) if pd.notna(points_a_value) else np.nan,
        "points_B": float(points_b_value) if pd.notna(points_b_value) else np.nan,
        "points_diff": float(points_diff_value) if pd.notna(points_diff_value) else np.nan,
        "points_missing_A": int(points_missing_a or pd.isna(points_a_value)),
        "points_missing_B": int(points_missing_b or pd.isna(points_b_value)),
        "age_A": age_a,
        "age_B": age_b,
        "age_diff": age_diff_value,
        "age_missing_A": age_missing_a,
        "age_missing_B": age_missing_b,
        "winner_name_raw": winner_name,
        "loser_name_raw": loser_name,
    }


def build_dataset(matches: pd.DataFrame, rankings: pd.DataFrame, players_lookup: PlayerLookup, limit: int | None = None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in matches.itertuples(index=False, name="MatchRow"):
        if limit is not None and len(records) >= limit:
            break
        record = canonicalize_ab(row._asdict(), rankings, players_lookup)
        records.append(record)
    df = pd.DataFrame(records)
    return add_one_hot_features(df)


def add_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    surfaces = ["Hard", "Clay", "Grass", "Carpet"]
    for surface in surfaces:
        df[f"surface_{surface}"] = (df["surface"].fillna("").eq(surface)).astype(int)

    target_rounds = ["F", "SF", "QF", "R16", "R32", "R64", "R128"]
    df["round_clean"] = df["round"].fillna("")
    for rnd in target_rounds:
        df[f"round_{rnd}"] = (df["round_clean"] == rnd).astype(int)
    df["round_other"] = (~df["round_clean"].isin(target_rounds) & df["round_clean"].ne("" )).astype(int)
    df = df.drop(columns=["round_clean"])
    return df


def describe_dataframe(df: pd.DataFrame, extra_lines: list[str] | None = None) -> str:
    lines = [
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns",
        "NaN ratio per column:",
    ]
    nan_ratios = df.isna().mean().sort_values(ascending=False)
    for col, ratio in nan_ratios.items():
        lines.append(f"  - {col}: {ratio:.1%}")
    flag_groups = [
        ("rank_missing_A", "rank_missing_A"),
        ("rank_missing_B", "rank_missing_B"),
        ("points_missing_A", "points_missing_A"),
        ("points_missing_B", "points_missing_B"),
        ("age_missing_A", "age_missing_A"),
        ("age_missing_B", "age_missing_B"),
    ]
    existing_flags = [name for name, _ in flag_groups if name in df.columns]
    if existing_flags:
        lines.append("Flagged missing ratios:")
        for name, label in flag_groups:
            if name in df.columns:
                ratio = float(df[name].mean()) if len(df) else 0.0
                lines.append(f"  - {label}: {ratio:.1%}")
    if "best_of_inferred" in df.columns:
        inferred_count = int(df["best_of_inferred"].sum())
        lines.append(f"best_of_inferred == 1 rows: {inferred_count}")
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines)


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def run(
    data_root: Path,
    years: list[int] | None,
    include_all_years: bool,
    limit: int | None,
    out_path: Path,
    min_year: int | None = None,
    max_year: int | None = None,
    add_recent_form: bool = False,
    lookback_matches: int = 20,
    min_recent_matches: int = 10,
) -> pd.DataFrame:
    if years and include_all_years:
        raise ValueError("Choisir --years ou --all-years, pas les deux.")
    if add_recent_form and min_recent_matches > lookback_matches:
        raise ValueError("--min-recent-matches doit être inférieur ou égal à --lookback-matches")
    hub = DataHub(data_root)
    _, players_lookup = prepare_players(hub.load_players())
    rankings_df = prepare_rankings(hub.load_rankings())

    if include_all_years or years is None or add_recent_form:
        matches_history = hub.load_matches()
    else:
        matches_history = hub.load_matches(years=years)

    matches_history = matches_history.copy()
    matches_history["tourney_date"] = pd.to_datetime(matches_history["tourney_date"], errors="coerce")

    if years:
        target_years = set(years)
        matches_df = matches_history[matches_history["tourney_date"].dt.year.isin(target_years)].copy()
    else:
        matches_df = matches_history.copy()

    if min_year is not None or max_year is not None:
        mask = pd.Series(True, index=matches_df.index)
        years_series = matches_df["tourney_date"].dt.year
        if min_year is not None:
            mask &= years_series >= min_year
        if max_year is not None:
            mask &= years_series <= max_year
        matches_df = matches_df[mask]

    matches_df = matches_df.sort_values("tourney_date", na_position="last").reset_index(drop=True)
    dataset = build_dataset(matches_df, rankings_df, players_lookup, limit=limit)

    extra_lines: list[str] = []
    if add_recent_form:
        if matches_df.empty:
            matches_for_form = matches_df
        else:
            max_date = matches_df["tourney_date"].max()
            matches_for_form = matches_history[matches_history["tourney_date"] <= max_date].copy()
        dataset = add_recent_form_features(
            matches_raw=matches_for_form,
            dataset_ab=dataset,
            lookback_matches=lookback_matches,
            min_matches=min_recent_matches,
        )
        recent_diff_cols = [
            f"win_rate_{lookback_matches}_diff",
            f"first_in_pct_{lookback_matches}_diff",
            f"first_won_pct_{lookback_matches}_diff",
            f"second_won_pct_{lookback_matches}_diff",
            f"aces_per_SvGm_{lookback_matches}_diff",
            f"df_per_SvGm_{lookback_matches}_diff",
            f"win_rate_surface_{lookback_matches}_diff",
        ]
        available_diffs = [col for col in recent_diff_cols if col in dataset.columns]
        if available_diffs:
            extra_lines.append("Recent form coverage:")
            total_rows = len(dataset)
            for col in available_diffs:
                coverage = float(dataset[col].notna().mean()) if total_rows else 0.0
                extra_lines.append(f"  - {col}: {coverage:.1%} non-NaN")
        if "recent_form_missing_A" in dataset.columns:
            extra_lines.append(f"recent_form_missing_A == 1 rows: {int(dataset['recent_form_missing_A'].sum())}")
        if "recent_form_missing_B" in dataset.columns:
            extra_lines.append(f"recent_form_missing_B == 1 rows: {int(dataset['recent_form_missing_B'].sum())}")

    save_dataset(dataset, out_path)
    print(describe_dataframe(dataset, extra_lines=extra_lines if extra_lines else None))
    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build A vs B dataset for TieBreaker AI")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Racine des données (défaut: data)")
    parser.add_argument("--years", type=int, nargs="*", help="Années à inclure (ex: 2023 2024)")
    parser.add_argument("--all-years", action="store_true", help="Inclure toutes les années disponibles")
    parser.add_argument("--min-year", type=int, help="Filtrer les matches à partir de cette année incluse")
    parser.add_argument("--max-year", type=int, help="Filtrer les matches jusqu'à cette année incluse")
    parser.add_argument("--limit", type=int, help="Limiter le nombre de matches traités (dev rapide)")
    parser.add_argument("--out", type=Path, default=Path("data/processed/dataset_outcome.parquet"), help="Fichier de sortie Parquet")
    parser.add_argument("--add-recent-form", action="store_true", help="Ajouter les features de forme récente (lookback par joueur)")
    parser.add_argument("--lookback-matches", type=int, default=20, help="Nombre de matches utilisés pour la forme récente (défaut: 20)")
    parser.add_argument("--min-recent-matches", type=int, default=10, help="Nombre minimum de matches requis pour calculer la forme (défaut: 10)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(
            data_root=args.data_root,
            years=args.years,
            include_all_years=args.all_years,
            limit=args.limit,
            out_path=args.out,
            min_year=args.min_year,
            max_year=args.max_year,
            add_recent_form=args.add_recent_form,
            lookback_matches=args.lookback_matches,
            min_recent_matches=args.min_recent_matches,
        )
    except Exception as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
