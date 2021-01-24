import os
import time

import yaml

os.makedirs("logs", exist_ok=True)
REPORT_FILE_NAME = f"logs/{round(time.time())}.yml"


class Reporter:
    report = {
        "runtime": 0,
        "dataset": {"hls": 0, "non_hls": 0, "total": 0},
        "train": {"hls": 0, "non_hls": 0, "total": 0},
        "test": {"hls": 0, "non_hls": 0, "total": 0},
        "result": {},
    }

    @classmethod
    def save(cls):
        with open(REPORT_FILE_NAME, "wt", encoding="utf-8") as f:
            yaml.dump(cls.report, f, sort_keys=False)
        print()
        print(yaml.dump(cls.report, sort_keys=False))
