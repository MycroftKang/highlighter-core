import time

import pandas as pd
from highlighter.utils.fetch import TwitchCrawler

CLIENT_ID = ""
BEARER_TOKEN = ""

targets = [
    "850693720",
    "840859405",
    "851421431",
    "856281191",
    "858487421",
    "857359079",
    "859685621",
    "860859118",
    "863268787",
    "863268787",
    "850266962",
    "849302446",
    "848944673",
    "848395376",
    "847271566",
    "846697408",
    "846111897",
    "842681261",
    "842034638",
    "841476435",
    "860859118",
    "865042827",
]

crawler = TwitchCrawler(CLIENT_ID, BEARER_TOKEN)
s = time.time()

for i, vid in enumerate(targets):
    print(f"==== {vid} ====")
    vlen = crawler.get_video_duration(vid)

    s2 = time.time()
    df: pd.DataFrame = crawler.get_chats(vid, vlen, worker=15)
    e2 = time.time()

    df.to_csv(
        f"twitch-data/chats/chats-{vid}-{vlen}.csv",
        index=False,
        columns=["time", "username", "chat"],
    )

    with open("log.txt", "at", encoding="utf-8") as f:
        f.write(f"{vid} ({vlen}): {e2 - s2}\n")

    print(f":: {(i+1)/len(targets)}")
print(time.time() - s)
