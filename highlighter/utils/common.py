import logging

import pandas as pd

log = logging.getLogger(__name__)
COLUMNS = {"num": float, "len": float, "hl": int}


def get_empty_dataframe(columns: list):
    gdf = pd.DataFrame(columns=columns)
    for c in columns:
        gdf[c] = gdf[c].astype(COLUMNS[c])
    return gdf


def get_fv_df(vlen, df: pd.DataFrame, columns_to_norm, fv_fn, win_size):
    df = (
        df.assign(
            chat=lambda x: x.chat.apply(lambda t: len(t)),
            range=pd.cut(df["time"], range(0, vlen, win_size)),
        )
        .groupby("range")
        .apply(fv_fn)
    )

    df[columns_to_norm] = df[columns_to_norm].apply(lambda x: x / x.max())

    return df
