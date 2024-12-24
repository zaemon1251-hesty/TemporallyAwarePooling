from dataset import CommentaryClipsForDiffEstimation
import polars as pl
import numpy as np
import json
from typing import Callable
from sklearn.metrics import confusion_matrix
from tap import Tap
from scipy.stats import lognorm, expon, gamma
import scipy
import numpy


class MainV2Argument(Tap):
    path: str = "./Benchmarks/TemporallyAwarePooling/data"
    fps: int = 1

    split: str = "test"

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
        "/Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/Additional_Info_Ratios__Before_and_After.csv"
    )
    action_window_size: float = 15
    addinfo_force: bool = False

    seed: int = 12

    def configure(self):
        self.add_argument("--lognorm_params", type=json.loads, required=False)
        self.add_argument("--gamma_params", type=json.loads, required=False)
        self.add_argument("--expon_params", type=json.loads, required=False)


def preprocess_action_df(spotting_df: pl.DataFrame) -> pl.DataFrame:
    # " - "で2分割（固定長2パーツ）
    split_cols = pl.col("gameTime").str.split_exact(" - ", 1)

    half_col = split_cols.struct.field("field_0").cast(pl.Float32).alias("half")
    game_time_str = split_cols.struct.field("field_1")

    # コロンの数をカウント
    colon_count = game_time_str.str.count_matches(":")

    # "HH:MM:SS"用にコロン2つの場合のsplit(3パーツ)
    three_parts = game_time_str.str.split_exact(":", 2)
    # "MM:SS"用にコロン1つの場合のsplit(2パーツ)
    two_parts = game_time_str.str.split_exact(":", 1)

    # colon_countに基づいて時間を計算
    time_col = (
        pl.when(colon_count == 2)
        .then(  # HH:MM:SS
            (three_parts.struct.field("field_1").cast(pl.Int32) * 60)
            + three_parts.struct.field("field_2").cast(pl.Int32)
        )
        .otherwise(  # MM:SS
            (two_parts.struct.field("field_0").cast(pl.Int32) * 60)
            + two_parts.struct.field("field_1").cast(pl.Int32)
        )
        .alias("time")
    )

    return spotting_df.with_columns(
        [half_col, time_col, pl.col("game").str.strip_chars_end("/").alias("game")]
    )


def to_gametime(half, seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milsec = (seconds - int(seconds)) * 100
    return f"{int(half)} - {int(minutes):02d}:{int(seconds):02d}.{int(milsec):02d}"


class SpottingModel:
    def __init__(self, args: MainV2Argument):
        self.args = args
        self.label_space = [0, 1]  # 映像の説明, 付加的情報
        self.label_prob = [0.82, 0.18]  # 全体のラベル割合分布

        self.mean_silence_sec = args.mean_silence_sec
        self.timing_algo = args.timing_algo
        self.label_algo = args.label_algo

        self.action_df = pl.read_csv(args.action_spotting_label_csv)
        self.action_df = preprocess_action_df(self.action_df)
        self.action_rate_df = pl.read_csv(args.action_rate_csv)
        self.action_window_size = args.action_window_size
        self.addinfo_force = args.addinfo_force

        self.lognorm_params = args.lognorm_params
        self.gamma_params = args.gamma_params
        self.expon_params = args.expon_params

    def __call__(self, previous_t, game=None, half=None, target_ts=None):
        if target_ts is not None:
            # teacher forcing for debug label prediciton
            next_ts = target_ts
        else:
            next_ts = self._next_ts(previous_t)

        next_label = self._next_label(game, half, next_ts)
        return (next_ts, next_label)

    def _next_ts(self, previous_t):
        if self.timing_algo == "constant":
            next_ts = previous_t + self.mean_silence_sec
        elif self.timing_algo == "lognorm":
            next_ts = (
                lognorm.rvs(
                    s=self.lognorm_params["shape"],
                    loc=self.lognorm_params["loc"],
                    scale=self.lognorm_params["scale"],
                )
                + previous_t
            )
        elif self.timing_algo == "gamma":
            next_ts = (
                gamma.rvs(
                    self.gamma_params["shape"],
                    scale=self.gamma_params["scale"],
                    loc=self.gamma_params["loc"],
                )
                + previous_t
            )
        elif self.timing_algo == "expon":
            next_ts = (
                expon.rvs(
                    scale=self.expon_params["scale"], loc=self.expon_params["loc"]
                )
                + previous_t
            )
        return next_ts

    def _next_label(self, game, half, next_t):
        # game, half, next_t が入力として必要
        assert isinstance(game, str)
        assert half in [1, 2]
        assert isinstance(next_t, int) or isinstance(next_t, float)

        if self.label_algo == "constant":
            next_label = np.random.choice(self.label_space, p=self.label_prob)
        elif self.label_algo == "action_spotting":
            label_result = self.action_df.filter(
                (self.action_df["game"] == game)
                & (self.action_df["half"] == half)
                & (self.action_df["time"] <= next_t + self.action_window_size)
                & (self.action_df["time"] >= next_t - self.action_window_size)
            )

            if len(label_result) == 0:
                label_prob = self.label_prob
            else:
                # polars 行アクセス
                # 最も self.action_df["time"]とnext_tが近い行を取得
                nearest_action_row = label_result.row(0, named=True)
                nearest_label = nearest_action_row["label"]
                is_before = next_t < nearest_action_row["time"]
                action_rate_result = self.action_rate_df.filter(
                    (self.action_rate_df["label"] == nearest_label)
                )

                if len(action_rate_result) == 0:
                    label_prob = self.label_prob
                else:
                    # labelの 前(rate_before) or 後(rate_after) の付加的情報の割合
                    col_suffix = "before" if is_before else "after"
                    addinfo_rate = action_rate_result.row(0, named=True)[
                        f"rate_{col_suffix}"
                    ]
                    assert 0 <= addinfo_rate <= 1

                    # 付加的情報の割合が高い(アクション,前後)場合、付加的情報とする
                    if args.addinfo_force and addinfo_rate > 0.18:
                        addinfo_rate = 1.0
                    label_prob = [1 - addinfo_rate, addinfo_rate]

            # ラベルを生成
            next_label = np.random.choice(self.label_space, p=label_prob)
        return next_label


def evaluate_diff_and_label(dataset, predict_model: Callable):
    result_dict = {
        "metadata": {
            "predict_ts": [],
            "target_ts": [],
            "predict_label": [],
            "target_label": [],
            "diff": [],
        },
        "content": [],
    }
    for game, half, previous_ts, target_ts, target_label in dataset:
        if args.ignore_under_1sec and (target_ts - previous_ts < 1):
            continue

        predict_ts, predict_label = predict_model(previous_ts, game, half)

        diff = (predict_ts - target_ts) ** 2

        result_dict["metadata"]["predict_ts"].append(predict_ts)
        result_dict["metadata"]["target_ts"].append(target_ts)
        result_dict["metadata"]["predict_label"].append(int(predict_label))
        result_dict["metadata"]["target_label"].append(int(target_label))
        result_dict["metadata"]["diff"].append(int(diff))

        result_dict["content"].append(
            {
                "game": game,
                "half": half,
                "previsou_ts": previous_ts,
                "previous_end_time": to_gametime(half, previous_ts),
                "predict_start_time": to_gametime(half, predict_ts),
                "predict_label": (
                    "付加的情報の提供" if int(predict_label) == 1 else "映像の説明"
                ),
            }
        )
    result_dict["content"] = sorted(
        result_dict["content"],
        key=lambda x: (x["game"], x["half"], x["previsou_ts"]),
    )

    # timing: calculate diff average
    diff_average = np.mean(result_dict["metadata"]["diff"])
    print(f"diff_average: {diff_average}")

    # label: confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        result_dict["metadata"]["target_label"],
        result_dict["metadata"]["predict_label"],
    ).ravel()
    print(f"confusion matrix: {tn=} {fp=} {fn=} {tp=}")
    ## calculate label F1
    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    f1_score = 2 * pr * re / (pr + re)
    print(f"label_accuracy: {(tp + tn) / (tp + tn + fp + fn):.3%}")
    print(f"label_precision: {pr:.3%}")
    print(f"label_recall: {re:.3%}")
    print(f"label_f1: {f1_score:.3%}")

    # save result dict json
    del result_dict["metadata"]
    result_dict["evaluation"] = {
        "diff_average": float(diff_average),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "label_accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "label_precision": float(pr),
        "label_recall": float(re),
        "label_f1": float(f1_score),
    }

    # 保存する
    with open(
        f"logs/spotting-result-{args.label_algo}-{args.timing_algo}.json", "w"
    ) as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    return


if __name__ == "__main__":
    args = MainV2Argument().parse_args()

    # set random seed
    args.lognorm_params["random_state"] = args.seed
    args.gamma_params["random_state"] = args.seed
    args.expon_params["random_state"] = args.seed
    numpy.random.seed(args.seed)

    dataset = CommentaryClipsForDiffEstimation(
        path=args.path,
        split=args.split,
        prev_ts_col="end",
        ts_col="start",
        label_col="付加的情報か",
    )

    print(f"len(dataset_Test): {len(dataset)}")

    spotting_model = SpottingModel(args)

    # 簡単な調査として、action_df の game と dataset の game がどれだけ一致しているかを調べる
    print(
        f"action_df と datasetの game が一致してる数: {len(set(spotting_model.action_df['game'].to_list()) & set(dataset.listGames))}"
    )

    evaluate_diff_and_label(dataset, spotting_model.__call__)
