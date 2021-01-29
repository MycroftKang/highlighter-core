import datetime
import os
import sys

from highlighter.utils.load import DataSetLoader
from highlighter.utils.predict import enumerate_fvs_df

if "-f" in sys.argv and len(sys.argv) > sys.argv.index("-f") + 1:
    target = sys.argv[sys.argv.index("-f") + 1]
else:
    target = "chats-840859405-11853.csv"


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
