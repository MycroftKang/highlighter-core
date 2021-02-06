import logging
import os
import time
from typing import List

import pandas as pd
import pygit2
from sklearn.model_selection import train_test_split

from . import common
from .load import VideoDataSet
from .report import Reporter

log = logging.getLogger(__name__)


def get_twitch_data_commit_sha():
    repo = pygit2.Repository(".git")
    sm: pygit2.Submodule = repo.lookup_submodule("twitch-data")
    return sm.head_id


def get_feature_vec(chats: pd.DataFrame, hls: pd.DataFrame):
    """
    return (num, len, hl)
    """

    if not chats.empty:
        start = chats.range.iloc[0].left
        end = chats.range.iloc[0].right
        chat_len = chats["chat"].mean(skipna=True)
        hl = int(
            hls.apply(
                lambda x: max(start, x["start"]) < min(end, x["end"]), axis=1
            ).any()
        )
    else:
        chat_len = 0
        hl = 0

    return pd.Series(
        [len(chats), chat_len, hl],
        index=["num", "len", "hl"],
    )


def to_data_frame(vds: List[VideoDataSet], win_size, use_cache=True):
    data_sha = get_twitch_data_commit_sha()
    cache_file = f"cache/cached_fvdf_{win_size}_{data_sha}.pkl"
    if use_cache and os.path.isfile(cache_file):
        gdf = pd.read_pickle(cache_file)
        print(f"Using cached {cache_file}")
    else:
        gdf = common.get_empty_dataframe(["num", "len", "hl"])
        for v in vds:
            s = time.time()
            temp = common.get_fv_df(
                v.vlen,
                v.chats,
                ["num", "len"],
                lambda _df: get_feature_vec(_df, v.hls),
                win_size,
            )
            validate_highlights(temp, v.hls, win_size)
            gdf = gdf.append(temp, ignore_index=True)
            print(f"Generated vector of {v.vid} in {round(time.time() - s, 2)}s")
        os.makedirs("cache", exist_ok=True)
        gdf.to_pickle(cache_file)
    log.debug(gdf)
    return gdf.loc[gdf["hl"] == 1], gdf.loc[gdf["hl"] == 0]


def validate_highlights(df: pd.DataFrame, hls: pd.DataFrame, win_size):
    for row in hls.itertuples():
        s = row.start
        e = row.end
        mask = df.apply(
            lambda x: max(s, x.name.left) < min(e, x.name.right), axis=1
        ).values
        if list(mask).count(True) > 1:
            df.loc[mask & (df.index != df.loc[mask]["num"].idxmax()), "hl"] = 0
        elif list(mask).count(True) == 0:
            print(f"{s}~{e} invalid!")


def get_dftrain_and_dfeval(
    vds: List[VideoDataSet], win_size=25, test_size=0.3, non_hls_size=1, use_cache=True
):
    hldf, non_hldf = to_data_frame(vds, win_size, use_cache=use_cache)
    log.debug(f"hldf: {hldf}")
    log.debug(f"non_hldf: {non_hldf}")

    if non_hls_size != 0:
        non_hldf = non_hldf.sample(n=min(len(hldf) * non_hls_size, len(non_hldf)))

    Reporter.report["dataset"] = {
        "hls": len(hldf),
        "non_hls": len(non_hldf),
        "total": len(hldf) + len(non_hldf),
    }

    hl_train, hl_test = train_test_split(hldf, test_size=test_size)
    non_hl_train, non_hl_test = train_test_split(non_hldf, test_size=test_size)

    train = hl_train.append(non_hl_train, ignore_index=True)
    test = hl_test.append(non_hl_test, ignore_index=True)

    Reporter.report["train"] = {
        "hls": len(hl_train),
        "non_hls": len(non_hl_train),
        "total": len(train),
    }
    Reporter.report["test"] = {
        "hls": len(hl_test),
        "non_hls": len(non_hl_test),
        "total": len(test),
    }

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    return train, test
