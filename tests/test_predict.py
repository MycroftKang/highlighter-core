from unittest import TestCase

import pandas as pd
from highlighter.predict import Predictor
from highlighter.utils.load import DataSetLoader


class TestPredictor(TestCase):
    def test_predict(self):
        predictor = Predictor()
        df = pd.DataFrame({"num": [0.8, 0.2], "len": [0.32, 0.65]})
        result = predictor.predict(df)

        self.assertEqual(result.columns.to_list(), ["class", "probability"])
        self.assertEqual(result.shape, (2, 2))

    def test_extract_range(self):
        predictor = Predictor()
        dfs = [
            pd.DataFrame(
                {
                    "class": [1, 1, 1, 1, 1],
                    "probability": [0.52, 0.65, 0.58, 0.76, 0.89],
                },
                index=[56, 57, 59, 62, 78],
            ),
            pd.DataFrame(
                {
                    "class": [1, 1, 1, 1, 1],
                    "probability": [0.52, 0.65, 0.58, 0.76, 0.89],
                },
                index=[56, 57, 59, 60, 61],
            ),
            pd.DataFrame(
                {
                    "class": [1, 1, 1, 1, 1, 1],
                    "probability": [0.52, 0.65, 0.58, 0.76, 0.89, 0.54],
                },
                index=[56, 57, 59, 60, 61, 78],
            ),
        ]

        expected = [
            [(1375, 1500), (1525, 1575), (1925, 1975)],
            [(1375, 1550)],
            [(1375, 1550), (1925, 1975)],
        ]

        for i, df in enumerate(dfs):
            with self.subTest(i=i):
                result = predictor.extract_range(df)
                self.assertIsInstance(result[0][0], int)
                self.assertEqual([x[:2] for x in result], expected[i])

    def test_get_highlights(self):
        predictor = Predictor()
        vcd = DataSetLoader().load_chats_by_vid("840859405")
        result = predictor.get_highlight_ranges(vcd)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(type(result[0]), tuple)
        self.assertEqual(len(result), 3)
