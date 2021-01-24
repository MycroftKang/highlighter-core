import logging

import pandas as pd

log = logging.getLogger(__name__)
COLUMNS = {"num": float, "len": float, "hl": int}


def get_empty_dataframe(columns: list):
    gdf = pd.DataFrame(columns=columns)
    for c in columns:
        gdf[c] = gdf[c].astype(COLUMNS[c])
    return gdf


def get_fv_df(vlen, df, columns, columns_to_norm, fv_fn, win_size):
    cur = 0
    ls = []
    while cur < vlen:
        ls.append(fv_fn(df, cur, cur + win_size))
        cur += win_size
    temp_df = pd.DataFrame(ls, columns=columns)
    temp_df[columns_to_norm] = temp_df[columns_to_norm].apply(lambda x: x / x.max())
    return temp_df
