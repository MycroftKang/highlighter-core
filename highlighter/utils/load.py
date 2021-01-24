import logging
import os
import re

import pandas as pd

log = logging.getLogger(__name__)


class VideoBaseData:
    def __init__(self, vid, vlen) -> None:
        self.vid = vid
        self.vlen = float(vlen)


class VideoChatsData(VideoBaseData):
    def __init__(self, vid, vlen, chats) -> None:
        super().__init__(vid, vlen)
        self.chats = chats


class VideoHlsData(VideoBaseData):
    def __init__(self, vid, vlen, hls) -> None:
        super().__init__(vid, vlen)
        self.hls = hls


class VideoDataSet(VideoChatsData, VideoHlsData):
    def __init__(self, vid, vlen, chats, hls) -> None:
        self.vid = vid
        self.vlen = float(vlen)
        self.chats = chats
        self.hls = hls


class DataSetLoader:
    def __init__(
        self, chats_dir="twitch-data/chats", hls_dir="twitch-data/hls"
    ) -> None:
        self.chats_dir = chats_dir
        self.hls_dir = hls_dir

    def load_dataset(self, name=None):
        """
        return {'vid':(vlen, DataFrame), ...}
        """
        dataset = []
        for (dirpath, _, filenames) in os.walk(self.chats_dir):
            log.debug(filenames)
            for fn in filenames:
                if (name != None) and fn != name:
                    continue
                m = re.search(r"chats-(?P<vid>\d+)-(?P<vlen>(?:\d+.)?\d+).csv", fn)
                if m:
                    meta = m.groupdict()
                else:
                    continue
                dataset.append(
                    VideoDataSet(
                        meta["vid"],
                        meta["vlen"],
                        pd.read_csv(f"{dirpath}/{fn}"),
                        pd.read_csv(f"{self.hls_dir}/{fn.replace('chats-', 'hls-')}"),
                    )
                )
                print(f"Read {dirpath}/{fn}")
        return dataset

    def load_chats(self, name=None):
        """
        return {'vid':(vlen, DataFrame), ...}
        """
        chats = []
        for (dirpath, _, filenames) in os.walk(self.chats_dir):
            log.debug(filenames)
            for fn in filenames:
                if (name != None) and fn != name:
                    continue
                m = re.search(r"chats-(?P<vid>\d+)-(?P<vlen>(?:\d+.)?\d+).csv", fn)
                if m:
                    meta = m.groupdict()
                else:
                    continue
                chats.append(
                    VideoChatsData(
                        meta["vid"],
                        float(meta["vlen"]),
                        chats=pd.read_csv(f"{dirpath}/{fn}"),
                    )
                )
                print(f"Read {dirpath}/{fn}")
        return chats

    def load_hls(self):
        """
        return {'vid':(vlen, DataFrame), ...}
        """
        hls = []
        for (dirpath, _, filenames) in os.walk(self.hls_dir):
            log.debug(filenames)
            for fn in filenames:
                m = re.search(r"hls-(?P<vid>\d+)-(?P<vlen>(?:\d+.)?\d+).csv", fn)
                if m:
                    meta = m.groupdict()
                else:
                    continue
                hls.append(
                    VideoHlsData(
                        meta["vid"],
                        float(meta["vlen"]),
                        hls=pd.read_csv(f"{dirpath}/{fn}"),
                    )
                )
                print(f"Read {dirpath}/{fn}")
        return hls
