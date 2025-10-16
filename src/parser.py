##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## parser
##

import pandas as pd

def parse_rank_date_col(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r'\.0$', '', regex=True)
    dt1 = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    dt2 = pd.to_datetime(s, errors="coerce")
    return dt1.fillna(dt2).dt.date