from dataset import CommentaryClipsForDiffEstimation
import pandas as pd
import numpy as np
import json
from typing import Callable
from sklearn.metrics import confusion_matrix
from tap import Tap
from scipy.stats import lognorm, expon, gamma


class MainV2Argument(Tap):
    path: str = "./Benchmarks/TemporallyAwarePooling/data"
    fps: int = 1

    # タイミング生成
    timing_algo: str = "constant"
    mean_silence_sec: float = (
        5.58  # 1秒以上の空白があるコメント集合における 平均的な発話間隔
    )
    lognorm_params: dict = {"shape": 2.2926, "loc": 0.0, "scale": 0.2688}
    gamma_params: dict = {"shape": 0.3283, "loc": 0.0, "scale": 6.4844}
    expon_params: dict = {"loc": 0.0, "scale": 2.1289}
    ignore_under_1sec: bool = False

    # ラベル生成
    label_algo: str = "constant"
    action_spotting_label_csv: str = (
        "/Users/heste/workspace/soccernet/sn-script/database/misc/soccernet_spotting_labels.csv"
    )
    action_rate_csv: str = (
        "/Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/Extracted_Action_Rates.csv"
    )
    action_window_size: int = 15

    def configure(self):
        self.add_argument("--lognorm_params", type=json.loads, required=False)
        self.add_argument("--gamma_params", type=json.loads, required=False)
        self.add_argument("--expon_params", type=json.loads, required=False)


def gametime_to_seconds(gametime):
    if gametime.count(":") == 2:
        gametime = ":".join(gametime.split(":")[:2])
    m, s = gametime.split(":")
    return int(m) * 60 + int(s)


def preprocess_action_df(spotting_df: pd.DataFrame):
    # 前処理
    spotting_df["half"] = spotting_df["gameTime"].str.split(" - ").str[0].astype(float)
    spotting_df["time"] = (
        spotting_df["gameTime"]
        .str.split(" - ")
        .str[1]
        .map(gametime_to_seconds)
        .astype(float)
    )
    spotting_df["game"] = spotting_df["game"].str.rstrip("/")
    return spotting_df


if __name__ == "__main__":
    args = MainV2Argument().parse_args()

    print(f"{args.as_dict()=}")

    dataset_Test = CommentaryClipsForDiffEstimation(
        path=args.path,
        split="test",
        prev_ts_col="end",
        ts_col="start",
        label_col="付加的情報か",
    )

    action_df = preprocess_action_df(pd.read_csv(args.action_spotting_label_csv))
    action_rate_df = pd.read_csv(args.action_rate_csv)

    def predict_diff_and_label(previous_t):
        mean_silence_sec = args.mean_silence_sec
        label_space = [0, 1]  # 映像の説明, 付加的情報
        label_prob = [0.82, 0.18]  # 全体のラベル割合分布

        if args.timing_algo == "constant":
            next_t = previous_t + mean_silence_sec
        elif args.timing_algo == "lognorm":
            next_t = previous_t + lognorm.rvs(
                s=args.lognorm_params["shape"],
                scale=args.lognorm_params["scale"],
            )
        elif args.timing_algo == "gamma":
            next_t = previous_t + gamma.rvs(
                args.gamma_params["shape"],
                scale=args.gamma_params["scale"],
            )
        elif args.timing_algo == "expon":
            next_t = previous_t + expon.rvs(scale=args.expon_params["scale"])
        else:
            raise ValueError("Invalid timing_algo")

        if args.label_algo == "constant":
            next_label = np.random.choice(label_space, p=label_prob)
        elif args.label_algo == "action_spotting":
            # ここにアクションスポッティングのアルゴリズムを実装する

            # game, half, time が入力として必要
            game = ""
            half = 1
            time = 0

            label_result = action_df[
                (action_df["game"] == game)
                & (action_df["half"] == half)
                & (action_df["time"] <= time + args.action_window_size)
                & (action_df["time"] >= time - args.action_window_size),
            ]
            if label_result.empty:
                label_prob = label_prob

            else:
                label = label_result.iloc[0]["label"]
                action_rate_result = action_rate_df[action_rate_df["label"] == label]

                if action_rate_result.empty:
                    label_prob = label_prob
                else:
                    rate = action_rate_result.iloc[0]["rate"]
                    label_prob = [1 - rate, rate]

            # ラベルを生成
            next_label = np.random.choice(label_space, p=label_prob)

        return (next_t, next_label)

    def evaluate_diff_and_label(dataset, predict_model: Callable):
        result_dict = {
            "diff": [],
            "label_same": [],
            "predict_label": [],
            "target_label": [],
        }
        for previous_frameid, target_frameid, target_label in dataset:
            if args.ignore_under_1sec and (target_frameid - previous_frameid < 1):
                continue

            next_frameid, predict_label = predict_model(previous_frameid)

            diff = (next_frameid - target_frameid) ** 2
            label_result = 1 if predict_label == target_label else 0

            result_dict["diff"].append(int(diff))
            result_dict["label_same"].append(int(label_result))
            result_dict["predict_label"].append(int(predict_label))
            result_dict["target_label"].append(int(target_label))

        # save result dict json
        with open("logs/baseline-spotting-result.json", "w") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

        # calculate diff average
        diff_average = np.mean(result_dict["diff"])
        print(f"diff_average: {diff_average}")

        # calculate label accuracy
        label_accuracy = np.mean(result_dict["label_same"])
        print(f"label_accuracy: {label_accuracy}")

        # confusion matrix
        matrix = confusion_matrix(
            result_dict["target_label"], result_dict["predict_label"]
        )
        print(f"confusion matrix: {matrix}")

        # calculate label F1
        tp = matrix[0][0]
        fn = matrix[0][1]
        fp = matrix[1][0]
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
        f1_score = 2 * pr * re / (pr + re)
        print(f"label_f1: {f1_score}")

        return

    evaluate_diff_and_label(dataset_Test, predict_diff_and_label)
