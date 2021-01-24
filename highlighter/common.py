import tensorflow as tf

NUMERIC_COLUMNS = ["num", "len"]

feature_columns = []
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(
        tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
    )
