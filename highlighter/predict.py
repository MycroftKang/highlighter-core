import pandas as pd
import tensorflow as tf

from .utils.load import VideoChatsData
from .utils.predict import get_fv_df_from_chats


def make_input_fn(data_df, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices(dict(data_df)).batch(batch_size)
        return ds

    return input_function


class Predictor:
    def __init__(self, model_dir, win_size=25) -> None:
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

    def get_hls(self, vd: VideoChatsData, num=3):
        fv_df = get_fv_df_from_chats(vd, self.win_size)
        df = self.predict(fv_df)
        return (
            df.loc[df["class"] == 1]
            .sort_values("probability", ascending=False)
            .head(num)
        )