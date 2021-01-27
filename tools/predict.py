import datetime
import os
import sys
from time import time
from typing import List

import pandas as pd

from highlighter.predict import Predictor
from highlighter.train import Trainer
from highlighter.utils.load import DataSetLoader, VideoChatsData
from highlighter.utils.predict import enumerate_fvs_df

target = "chats-848395376-43020.0.csv" if len(sys.argv) < 2 else sys.argv[1]
chats = DataSetLoader().load_chats(target)


def get_hls_from_vds(vds: List[VideoChatsData], trainer, num=3):
    for vid, t in enumerate_fvs_df(vds, trainer.win_size):
        df = trainer.predict(t)
        top = (
            df.loc[df["class"] == 1]
            .sort_values("probability", ascending=False)
            .head(num)
        )
        yield vid, top


def print_hls(vid, top, win_size):
    print()
    ls = [f"Video ID: {vid}", "time | probability"]
    for i, row in top.iterrows():
        s = datetime.timedelta(seconds=i * win_size)
        e = datetime.timedelta(seconds=(i + 1) * win_size)
        ls.append(f"{s} ~ {e} | {row['probability']}")
    print("\n".join(ls))


def trainer_predict():
    trainer = Trainer()
    for vd in chats:
        top = trainer.get_hls(vd, 10)
        print_hls(vd.vid, top, trainer.win_size)


def predictor_predict():
    predictor = Predictor(r"models\1611472888")
    for vd in chats:
        top = predictor.get_hls(vd, 10)
        print_hls(vd.vid, top, predictor.win_size)


s = time()
trainer_predict()
print(time() - s)
s = time()
predictor_predict()
print(time() - s)
