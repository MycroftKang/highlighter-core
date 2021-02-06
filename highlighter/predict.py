import heapq
import os

import pandas as pd
import tensorflow as tf

from .utils.load import VideoChatsData
from .utils.path import MODULE_ROOT_PATH
from .utils.predict import get_fv_df_from_chats

DEFAULT_MODEL_DIR = os.path.join(MODULE_ROOT_PATH, "models")


class Predictor:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, win_size=25) -> None:
        self.win_size = win_size
        self.imported = tf.saved_model.load(model_dir)

    def predict(self, df: pd.DataFrame):
        examples = [
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        k: tf.train.Feature(float_list=tf.train.FloatList(value=[v]))
                        for k, v in zip(["num", "len"], [row.num, row.len])
                    }
                )
            ).SerializeToString()
            for row in df.itertuples()
        ]

        result = self.imported.signatures["predict"](
            examples=tf.constant(examples),
        )

        return pd.DataFrame(
            [
                [class_id[0], prob[class_id[0]]]
                for class_id, prob in zip(
                    result["class_ids"].numpy(), result["probabilities"].numpy()
                )
            ],
            columns=["class", "probability"],
        )

    def extract_range(self, df: pd.DataFrame):
        df_len = len(df)
        df_max_idx = df_len - 1
        win = df.index.tolist()
        out = []
        t = 0

        for i in range(df_len):
            if df_max_idx == i or win[i + 1] - win[i] > 2:
                rg = (
                    win[t] * self.win_size - 25,
                    (win[i] + 1) * self.win_size,
                    df["probability"].iloc[t],
                )

                t = i + 1

                out.append(rg)

        return out

    def get_highlight_ranges(self, vd: VideoChatsData, num=3):
        fv_df = get_fv_df_from_chats(vd, self.win_size)

        df = self.predict(fv_df)
        df = df.loc[df["class"] == 1]

        result = self.extract_range(df)

        if num != 0 and len(result) > num:
            result = [
                v
                for (_, v) in sorted(
                    heapq.nlargest(num, enumerate(result), lambda x: x[1][2])
                )
            ]

        return result
