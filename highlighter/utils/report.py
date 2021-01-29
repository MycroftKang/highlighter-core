import os
import time

import yaml

os.makedirs("logs", exist_ok=True)
REPORT_INIT_TIME = time.time()


class Reporter:
    report = {
        "runtime": 0,
        "dataset": {"hls": 0, "non_hls": 0, "total": 0},
        "train": {"hls": 0, "non_hls": 0, "total": 0},
        "test": {"hls": 0, "non_hls": 0, "total": 0},
        "result": {},
    }

    @classmethod
    def save(cls, file_name=None):
        if file_name is None:
            file_name = REPORT_INIT_TIME
        with open(f"logs/{file_name}.yml", "wt", encoding="utf-8") as f:
            yaml.dump(cls.report, f, sort_keys=False)
        print()
        print(yaml.dump(cls.report, sort_keys=False))
