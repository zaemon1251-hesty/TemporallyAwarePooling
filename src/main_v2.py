from dataset import CommentaryClipsForDiffEstimation

import numpy as np
import json
from typing import Callable
from sklearn.metrics import confusion_matrix
from tap import Tap


class MainV2Argument(Tap):
    path: str = "./Benchmarks/TemporallyAwarePooling/data"
    fps: int = 1
    mean_silence_sec: float = (
        5.58  # 1秒以上の空白があるコメント集合における 平均的な発話間隔
    )


if __name__ == "__main__":
    args = MainV2Argument().parse_args()

    dataset_Test = CommentaryClipsForDiffEstimation(
        path=args.path,
        split="test",
        prev_ts_col="end",
        ts_col="start",
        label_col="付加的情報か",
    )

    def predict_diff_and_label(previous_t):
        mean_silence_sec = args.mean_silence_sec
        fps = args.fps
        label_space = [0, 1]  # 映像の説明, 付加的情報
        label_prob = [0.82, 0.18]  # 全体のラベル割合分布

        next_t = previous_t + int(mean_silence_sec * fps)
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
            next_frameid, predict_label = predict_model(previous_frameid)
            diff = abs(next_frameid - target_frameid)
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
