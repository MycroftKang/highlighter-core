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
                        for k, v in row.items()
                    }
                )
            ).SerializeToString()
            for _, row in df.iterrows()
        ]

        result = self.imported.signatures["predict"](
            examples=tf.constant(examples),
        )

        return pd.DataFrame(
            [
                [class_id[0], result["probabilities"][idx][class_id[0]].numpy()]
                for idx, class_id in enumerate(result["class_ids"].numpy())
            ],
            columns=["class", "probability"],
        )

    def extract_range(self, df: pd.DataFrame, probability=False):
        df_len = len(df)
        df_max_idx = df_len - 1
        win = df.index.tolist()
        out = []
        t = 0

        while df_len != t:
            i = t
            for i in range(i, df_len):
                if (i < df_max_idx) and (win[i + 1] - win[i] > 2):
                    break

            if probability:
                rg = (
                    win[t] * self.win_size - 25,
                    (win[i] + 1) * self.win_size,
                    df["probability"].iloc[t],
                )
            else:
                rg = (
                    win[t] * self.win_size - 25,
                    (win[i] + 1) * self.win_size,
                )

            out.append(rg)
            t = i + 1

        return out

    def get_highlight_ranges(self, vd: VideoChatsData, num=3, probability=False):
        fv_df = get_fv_df_from_chats(vd, self.win_size)

        df = self.predict(fv_df)
        df = df.loc[df["class"] == 1].sort_values("probability", ascending=False)
        result = []

        for i in range(num, len(df)):
            result = self.extract_range(
                df.iloc[:i].sort_index(), probability=probability
            )
            if len(result) == num:
                return result
        return result
