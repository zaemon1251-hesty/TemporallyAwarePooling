#!/bin/bash
# 沈黙時間 2.14 ← 平均的な発話間隔
# label_prob = [0.82, 0.18]  # 全体のラベル割合分布

path="./data"

python src/main_v2.py --path $path --mean_silence_sec 2.14

<< COMMENTOUT
test 集合での結果
diff_average: 2.60957
label_accuracy: 0.70294
confusion matrix: [
 [46129 10015]
 [10458  2317]
]
label_f1: 0.81839
COMMENTOUT
