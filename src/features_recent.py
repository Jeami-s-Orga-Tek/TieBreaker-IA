from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RecentFormConfig:
    lookback_matches: int
    min_matches: int


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().split()).casefold()


def _player_key(id_series: pd.Series, name_series: pd.Series) -> pd.Series:
    numeric_ids = pd.to_numeric(id_series, errors="coerce")
    keys = numeric_ids.apply(lambda x: f"id:{int(x)}" if pd.notna(x) else "")
    fallback_names = name_series.fillna("").map(_normalize_name)
    fallback = fallback_names.apply(lambda x: f"name:{x}" if x else "")
    combined = keys.where(keys != "", fallback)
    combined = combined.replace({"": np.nan})
    return combined


def _get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, numerator / denominator, np.nan)


def _prepare_player_rows(matches_raw: pd.DataFrame) -> pd.DataFrame:
    matches = matches_raw.copy()
    matches["tourney_date_ts"] = pd.to_datetime(matches["tourney_date"], errors="coerce").dt.normalize()
    matches = matches.dropna(subset=["tourney_date_ts"])
    surface_series = matches.get("surface")
    surface_str = surface_series.fillna("").astype(str).str.strip()
    surface_title = surface_str.where(surface_str == "", surface_str.str.title())
    surface_title = surface_title.replace("", pd.NA)
    matches["surface_clean"] = surface_title
    matches["surface_key"] = matches["surface_clean"].fillna("__MISSING__")

    winner_df = pd.DataFrame({
        "player_key": _player_key(matches.get("winner_id"), matches.get("winner_name")),
        "player_name": matches.get("winner_name").fillna(""),
        "tourney_date": matches["tourney_date_ts"],
        "surface": matches["surface_clean"],
        "surface_key": matches["surface_key"],
        "won": 1,
    })

    loser_df = pd.DataFrame({
        "player_key": _player_key(matches.get("loser_id"), matches.get("loser_name")),
        "player_name": matches.get("loser_name").fillna(""),
        "tourney_date": matches["tourney_date_ts"],
        "surface": matches["surface_clean"],
        "surface_key": matches["surface_key"],
        "won": 0,
    })

    stat_map = {
        "ace": ("w_ace", "l_ace"),
        "df": ("w_df", "l_df"),
        "svpt": ("w_svpt", "l_svpt"),
        "first_in": ("w_1stIn", "l_1stIn"),
        "first_won": ("w_1stWon", "l_1stWon"),
        "second_won": ("w_2ndWon", "l_2ndWon"),
        "sv_gms": ("w_SvGms", "l_SvGms"),
        "bp_saved": ("w_bpSaved", "l_bpSaved"),
        "bp_faced": ("w_bpFaced", "l_bpFaced"),
    }

    for target, (winner_col, loser_col) in stat_map.items():
        winner_df[target] = _get_numeric_series(matches, winner_col)
        loser_df[target] = _get_numeric_series(matches, loser_col)

    player_rows = pd.concat([winner_df, loser_df], ignore_index=True)
    player_rows = player_rows.dropna(subset=["player_key"])
    player_rows = player_rows.sort_values(["player_key", "tourney_date"]).reset_index(drop=True)
    player_rows["surface_key"] = player_rows["surface_key"].fillna("__MISSING__")
    return player_rows


def add_recent_form_features(
    matches_raw: pd.DataFrame,
    dataset_ab: pd.DataFrame,
    *,
    lookback_matches: int = 20,
    min_matches: int = 10,
) -> pd.DataFrame:
    config = RecentFormConfig(lookback_matches=lookback_matches, min_matches=min_matches)
    player_rows = _prepare_player_rows(matches_raw)
    if player_rows.empty:
        return dataset_ab.assign(
            **{
                f"win_rate_{config.lookback_matches}_diff": np.nan,
                f"first_in_pct_{config.lookback_matches}_diff": np.nan,
                f"first_won_pct_{config.lookback_matches}_diff": np.nan,
                f"second_won_pct_{config.lookback_matches}_diff": np.nan,
                f"aces_per_SvGm_{config.lookback_matches}_diff": np.nan,
                f"df_per_SvGm_{config.lookback_matches}_diff": np.nan,
                f"win_rate_surface_{config.lookback_matches}_diff": np.nan,
                "recent_form_missing_A": 1,
                "recent_form_missing_B": 1,
            }
        )

    group_players = player_rows.groupby("player_key", group_keys=False)

    lookback = config.lookback_matches
    min_p = config.min_matches

    player_rows[f"win_rate_{lookback}"] = (
        group_players["won"].rolling(window=lookback, min_periods=min_p).mean().reset_index(level=0, drop=True)
    )

    rolling_svpt = group_players["svpt"].rolling(window=lookback, min_periods=min_p).sum().reset_index(level=0, drop=True)
    rolling_first_in = group_players["first_in"].rolling(window=lookback, min_periods=min_p).sum().reset_index(level=0, drop=True)
    rolling_first_won = group_players["first_won"].rolling(window=lookback, min_periods=min_p).sum().reset_index(level=0, drop=True)
    rolling_second_won = group_players["second_won"].rolling(window=lookback, min_periods=min_p).sum().reset_index(level=0, drop=True)
    rolling_sv_gms = group_players["sv_gms"].rolling(window=lookback, min_periods=min_p).sum().reset_index(level=0, drop=True)
    rolling_ace = group_players["ace"].rolling(window=lookback, min_periods=min_p).sum().reset_index(level=0, drop=True)
    rolling_df = group_players["df"].rolling(window=lookback, min_periods=min_p).sum().reset_index(level=0, drop=True)

    player_rows[f"first_in_pct_{lookback}"] = _safe_divide(rolling_first_in, rolling_svpt)
    player_rows[f"first_won_pct_{lookback}"] = _safe_divide(rolling_first_won, rolling_first_in)
    player_rows[f"second_won_pct_{lookback}"] = _safe_divide(rolling_second_won, rolling_svpt - rolling_first_in)
    player_rows[f"aces_per_SvGm_{lookback}"] = _safe_divide(rolling_ace, rolling_sv_gms)
    player_rows[f"df_per_SvGm_{lookback}"] = _safe_divide(rolling_df, rolling_sv_gms)

    group_surface = player_rows.groupby(["player_key", "surface_key"], group_keys=False)
    player_rows[f"win_rate_surface_{lookback}"] = (
        group_surface["won"].rolling(window=lookback, min_periods=min_p).mean().reset_index(level=[0, 1], drop=True)
    )

    match_counts = group_players["won"].rolling(window=lookback, min_periods=1).count().reset_index(level=0, drop=True)
    player_rows["_matches_count"] = match_counts

    # Shift to avoid using current match data
    feature_cols = [
        f"win_rate_{lookback}",
        f"first_in_pct_{lookback}",
        f"first_won_pct_{lookback}",
        f"second_won_pct_{lookback}",
        f"aces_per_SvGm_{lookback}",
        f"df_per_SvGm_{lookback}",
    ]
    player_rows[feature_cols] = group_players[feature_cols].shift(1)

    player_rows[f"win_rate_surface_{lookback}"] = group_surface[f"win_rate_surface_{lookback}"].shift(1)
    player_rows["_matches_count"] = group_players["_matches_count"].shift(1)

    player_rows["recent_form_missing"] = (
        player_rows["_matches_count"].fillna(0) < min_p
    ).astype(int)

    player_rows = player_rows.drop(columns=["_matches_count"])

    player_features = player_rows[
        [
            "player_key",
            "tourney_date",
            "surface_key",
            f"win_rate_{lookback}",
            f"first_in_pct_{lookback}",
            f"first_won_pct_{lookback}",
            f"second_won_pct_{lookback}",
            f"aces_per_SvGm_{lookback}",
            f"df_per_SvGm_{lookback}",
            f"win_rate_surface_{lookback}",
            "recent_form_missing",
        ]
    ].copy()

    player_features["tourney_date"] = pd.to_datetime(player_features["tourney_date"], errors="coerce").dt.normalize()
    player_features["surface_key"] = player_features["surface_key"].astype(str)
    player_features = player_features.dropna(subset=["player_key", "tourney_date"])
    player_features = player_features.drop_duplicates(subset=["player_key", "tourney_date", "surface_key"])

    dataset = dataset_ab.copy()
    dataset["_tourney_date"] = pd.to_datetime(dataset["tourney_date"], errors="coerce").dt.normalize()
    surface_raw = dataset["surface"].fillna("").astype(str).str.strip()
    surface_norm = surface_raw.where(surface_raw == "", surface_raw.str.title())
    surface_norm = surface_norm.replace("", pd.NA)
    dataset["_surface_key"] = surface_norm.fillna("__MISSING__")
    dataset["_A_key"] = _player_key(dataset.get("A_player_id"), dataset.get("A_name"))
    dataset["_B_key"] = _player_key(dataset.get("B_player_id"), dataset.get("B_name"))

    base_features = [
        f"win_rate_{lookback}",
        f"first_in_pct_{lookback}",
        f"first_won_pct_{lookback}",
        f"second_won_pct_{lookback}",
        f"aces_per_SvGm_{lookback}",
        f"df_per_SvGm_{lookback}",
        f"win_rate_surface_{lookback}",
    ]

    features_a = player_features.rename(
        columns={
            "player_key": "_A_key",
            "tourney_date": "_tourney_date",
            "surface_key": "_surface_key",
            "recent_form_missing": "recent_form_missing_A",
            **{col: f"{col}_A" for col in base_features},
        }
    )

    dataset = dataset.merge(features_a, on=["_A_key", "_tourney_date", "_surface_key"], how="left")

    features_b = player_features.rename(
        columns={
            "player_key": "_B_key",
            "tourney_date": "_tourney_date",
            "surface_key": "_surface_key",
            "recent_form_missing": "recent_form_missing_B",
            **{col: f"{col}_B" for col in base_features},
        }
    )

    dataset = dataset.merge(features_b, on=["_B_key", "_tourney_date", "_surface_key"], how="left")

    for base_col in base_features:
        col_a = f"{base_col}_A"
        col_b = f"{base_col}_B"
        diff_col = f"{base_col}_diff"
        dataset[diff_col] = dataset[col_a] - dataset[col_b]

    dataset["recent_form_missing_A"] = dataset.get("recent_form_missing_A").fillna(1).astype(int)
    dataset["recent_form_missing_B"] = dataset.get("recent_form_missing_B").fillna(1).astype(int)

    drop_cols = [f"{base}_A" for base in base_features] + [f"{base}_B" for base in base_features]
    dataset = dataset.drop(columns=drop_cols, errors="ignore")
    dataset = dataset.drop(columns=["_A_key", "_B_key", "_surface_key", "_tourney_date"], errors="ignore")

    return dataset
