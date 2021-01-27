import datetime
import os
import sys

from highlighter.utils.load import DataSetLoader
from highlighter.utils.predict import enumerate_fvs_df

target = "chats-848395376-43020.0.csv" if len(sys.argv) < 2 else sys.argv[1]

chats = DataSetLoader().load_chats(target)
win_size = 25

print()
for vid, t in enumerate_fvs_df(chats, win_size):
    top = t.sort_values("num", ascending=False).head(10)
    print("Video ID: ", vid)
    print("time | num | len")
    for i, row in top.iterrows():
        s = datetime.timedelta(seconds=i * win_size)
        e = datetime.timedelta(seconds=(i + 1) * win_size)
        print(f"{s} ~ {e} | {row['num']} | {row['len']}")
    print()
