import asyncio
import datetime
import re

import aiohttp
import pandas as pd
import requests


class WorkerStorage:
    def __init__(self, vid, worker_count) -> None:
        self.vid = vid
        self.first_ids = {}
        self.fetch_data = {}
        self.worker_count = worker_count

    def store_first_id(self, wid, msgid):
        self.first_ids[wid] = msgid

    def store_fetch_data(self, wid, df):
        if wid in self.fetch_data:
            self.fetch_data[wid] = self.fetch_data[wid].append(df, ignore_index=True)
        else:
            self.fetch_data[wid] = df
            self.store_first_id(wid, df["id"].iloc[0])

    def is_finish(self, wid):
        if (wid + 1) == self.worker_count:
            return False

        if (wid + 1) not in self.first_ids:
            return False

        tar_id = self.first_ids[wid + 1]
        tar_df = self.fetch_data[wid]

        ils = tar_df.loc[tar_df["id"] == tar_id].index.tolist()
        if len(ils) != 0:
            self.fetch_data[wid] = tar_df.iloc[: ils[0]]
            return True
        else:
            return False

    def get_result(self):
        ls = [self.fetch_data[i] for i in sorted(self.fetch_data)]
        return pd.concat(ls)

    def get_store_fetch_data_fn(self, wid):
        def store_fn(df):
            return self.store_fetch_data(wid, df)

        return store_fn


class TwitchFetchWorker:
    def __init__(
        self, _id: int, client_id, offset, worker_storage: WorkerStorage, verbose=False
    ) -> None:
        self.id = _id
        self.vid = worker_storage.vid
        self.client_id = client_id
        self.offset = offset
        self.worker_storage = worker_storage
        self.store_fetch_data = worker_storage.get_store_fetch_data_fn(self.id)
        self.verbose = verbose

    def to_dataframe(self, data: list):
        temp = pd.DataFrame(data, columns=["id", "time", "username", "chat"])
        self.store_fetch_data(temp)

    def get_next_url(self, js):
        if self.worker_storage.is_finish(self.id):
            return None

        if "_next" in js:
            return f"https://api.twitch.tv/v5/videos/{self.vid}/comments?cursor={js['_next']}"
        else:
            return None

    async def download(self):
        headers = {
            "Accept": "application/vnd.twitchtv.v5+json",
            "Client-ID": self.client_id,
        }
        fetch_url = f"https://api.twitch.tv/v5/videos/{self.vid}/comments?content_offset_seconds={self.offset}"
        async with aiohttp.ClientSession(headers=headers) as session:
            while fetch_url is not None:
                async with session.get(fetch_url) as resp:
                    if resp.status != 200:
                        print(self.id, " Sleep ", resp.status)
                        asyncio.sleep(1)
                        continue
                    js = await resp.json()

                if self.verbose:
                    print(
                        self.id,
                        " ",
                        fetch_url.replace("https://api.twitch.tv/v5/videos", "")[:45],
                        " ",
                        resp.status,
                    )

                ls = [
                    [
                        c["_id"],
                        c["content_offset_seconds"],
                        c["commenter"]["name"],
                        c["message"]["body"],
                    ]
                    for c in js["comments"]
                ]
                self.to_dataframe(ls)
                fetch_url = self.get_next_url(js)

        print(
            f"Finished Worker {self.id} df: ",
            len(self.worker_storage.fetch_data[self.id]),
        )


class TwitchCrawler:
    def __init__(self, client_id, bearer_token) -> None:
        self.client_id = client_id
        self.bearer_token = bearer_token

    def duration_to_int(self, tstr):
        if m := re.match(
            r"(?:(?P<hour>\d+)h)?(?:(?P<min>\d+)m)?(?:(?P<sec>\d+)s)?", tstr
        ):
            a = m.groupdict()
            a = {k: int(v) if v is not None else 0 for k, v in a.items()}

            return datetime.timedelta(
                hours=a["hour"], minutes=a["min"], seconds=a["sec"]
            ).seconds
        else:
            raise ValueError

    def get_video_duration(self, vid):
        vidHeaders = {
            "Client-ID": self.client_id,
            "Authorization": "Bearer " + self.bearer_token,
        }
        vidParams = {"id": vid}

        res = requests.get(
            "https://api.twitch.tv/helix/videos", headers=vidHeaders, params=vidParams
        )

        res.raise_for_status()
        js = res.json()

        return self.duration_to_int(js["data"][0]["duration"])

    def get_chats(self, vid, vlen=None, worker=4, verbose=False):
        if vlen is None:
            vlen = self.get_video_duration(vid)
            print(vlen)

        storage = WorkerStorage(vid, worker)
        tasks = [
            TwitchFetchWorker(
                x, self.client_id, (vlen / worker) * x, storage, verbose
            ).download()
            for x in range(worker)
        ]

        asyncio.run(asyncio.wait(tasks))

        return storage.get_result()

    @staticmethod
    def get_twitch_token(client_id, client_secret):
        params = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": "clips:edit",  # may be added further later for analytics
        }

        r = requests.post("https://id.twitch.tv/oauth2/token", params=params)
        r.raise_for_status()

        bearer_token = r.json()["access_token"]

        return bearer_token
