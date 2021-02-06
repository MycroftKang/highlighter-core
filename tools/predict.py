import datetime
import sys
from time import time
from typing import List

from highlighter.utils.load import DataSetLoader, VideoChatsData
from highlighter.utils.predict import enumerate_fvs_df

if "-i" in sys.argv and len(sys.argv) > sys.argv.index("-i") + 1:
    target = sys.argv[sys.argv.index("-i") + 1]
else:
    target = "840859405"

vcd = DataSetLoader().load_chats_by_vid(target)


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


def print_hls_for_predictor(vid, top, win_size):
    print()
    ls = [f"Video ID: {vid}", "time | probability"]
    if len(top) == 0:
        print("None")
        return

    for st, et, pr in top:
        s = datetime.timedelta(seconds=st)
        e = datetime.timedelta(seconds=et)
        ls.append(f"{s} ~ {e} | {pr}")
    print("\n".join(ls))


def trainer_predict():
    from highlighter.train import Trainer

    trainer = Trainer()
    s = time()
    top = trainer.get_highlights(vcd, 10)
    print(time() - s)
    print_hls(vcd.vid, top, trainer.win_size)


def predictor_predict():
    from highlighter.predict import Predictor

    predictor = Predictor()
    s = time()
    hls = predictor.get_highlight_ranges(vcd, 10)
    print(time() - s)
    print_hls_for_predictor(vcd.vid, hls, predictor.win_size)


if "-t" in sys.argv:
    trainer_predict()
else:
    predictor_predict()
