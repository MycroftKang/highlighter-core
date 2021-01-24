import time

import pandas as pd
import tensorflow as tf

from .common import feature_columns
from .utils.load import DataSetLoader, VideoChatsData
from .utils.predict import get_fv_df_from_chats
from .utils.report import Reporter
from .utils.train import get_dftrain_and_dfeval


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


def make_predict_input_fn(data_df, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices(dict(data_df)).batch(batch_size)
        return ds

    return input_function


class Trainer:
    def __init__(self, win_size=25, non_hls_size=5, use_cache=True) -> None:
        self.linear_est = tf.estimator.LinearClassifier(
            feature_columns=feature_columns, model_dir="ckpts"
        )
        self.win_size = win_size
        self.non_hls_size = non_hls_size
        self.use_cache = use_cache

    def train(self, dftrain, y_train):
        train_input_fn = make_input_fn(dftrain, y_train)
        self.linear_est.train(train_input_fn)

    def evaluate(self, dfeval, y_eval):
        eval_input_fn = make_input_fn(dfeval, y_eval)
        return self.linear_est.evaluate(eval_input_fn)

    def predict(self, df: pd.DataFrame):
        input_fn = make_predict_input_fn(df)
        predicts = self.linear_est.predict(
            input_fn, predict_keys=["class_ids", "probabilities"]
        )

        return pd.DataFrame(
            [
                [pred["class_ids"][0], pred["probabilities"][pred["class_ids"]][0]]
                for pred in predicts
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

    def run(self, report=True, save_model=True):
        s = time.time()
        ds = DataSetLoader().load_dataset()
        dftrain, dfeval = get_dftrain_and_dfeval(
            ds,
            non_hls_size=self.non_hls_size,
            use_cache=self.use_cache,
            win_size=self.win_size,
        )
        y_train = dftrain.pop("hl")
        y_eval = dfeval.pop("hl")

        self.train(dftrain, y_train)
        result = self.evaluate(dfeval, y_eval)
        if save_model:
            self.save_model()
        if report:
            Reporter.report["runtime"] = round(time.time() - s, 2)
            Reporter.report["result"] = {str(k): float(v) for k, v in result.items()}
            Reporter.save()
        return result

    def save_model(self):
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            tf.feature_column.make_parse_example_spec(feature_columns)
        )
        estimator_path = self.linear_est.export_saved_model("models", serving_input_fn)
        print(estimator_path)
        return estimator_path
