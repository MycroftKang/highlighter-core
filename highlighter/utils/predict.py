import logging
from typing import List

import pandas as pd

from . import common
from .load import VideoChatsData

log = logging.getLogger(__name__)


def get_feature_vec(chats: pd.DataFrame, start: int, end: int):
    """
    return (num, len)
    """
    tar = chats[chats["time"].between(start, end)]
    logging.debug(tar)
    mean = tar["chat"].apply(lambda x: len(x)).mean(skipna=True) if not tar.empty else 0
    return len(tar), mean


def to_fv_df(vds: List[VideoChatsData], win_size=25):
    gdf = common.get_empty_dataframe(["num", "len"])
    for v in vds:
        temp = common.get_fv_df(
            v.vlen, v.chats, ["num", "len"], ["num", "len"], get_feature_vec, win_size
        )
        gdf = gdf.append(temp, ignore_index=True)
    log.debug(gdf)
    return gdf


def enumerate_fvs_df(vds: List[VideoChatsData], win_size=25):
    for v in vds:
        temp = common.get_fv_df(
            v.vlen, v.chats, ["num", "len"], ["num", "len"], get_feature_vec, win_size
        )
        yield (v.vid, temp)


def get_fv_df_from_chats(vd: VideoChatsData, win_size=25):
    temp = common.get_fv_df(
        vd.vlen, vd.chats, ["num", "len"], ["num", "len"], get_feature_vec, win_size
    )
    return temp
