import datetime
import sys
from time import time
from typing import List

from highlighter.utils.load import DataSetLoader, VideoChatsData
from highlighter.utils.predict import enumerate_fvs_df

if "-f" in sys.argv and len(sys.argv) > sys.argv.index("-f") + 1:
    target = sys.argv[sys.argv.index("-f") + 1]
else:
    target = "chats-840859405-11853.csv"

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
    from highlighter.train import Trainer

    trainer = Trainer()
    s = time()
    print(time() - s)
    for vd in chats:
        top = trainer.get_hls(vd, 10)
        print_hls(vd.vid, top, trainer.win_size)


def predictor_predict():
    from highlighter.predict import Predictor

    predictor = Predictor()
    s = time()
    for vd in chats:
        top = predictor.get_hls(vd, 10)
        print_hls(vd.vid, top, predictor.win_size)
    print(time() - s)


if "-t" in sys.argv:
    trainer_predict()
else:
    predictor_predict()
