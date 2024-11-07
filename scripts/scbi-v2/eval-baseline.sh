#!/bin/bash
# 沈黙時間 5.58 ← 1秒以上の空白があるコメント集合における平均的な発話間隔
# label_prob = [0.82, 0.18]  # 全体のラベル割合分布

path="./data"

python src/main_v2.py --path $path --mean_silence_sec 5.58

<< COMMENTOUT
test 集合での結果
diff_average: 5.2200
label_accuracy: 0.7027
confusion matrix: [
 [46116 10028]
 [10462  2313]
]
label_f1: 0.8182
COMMENTOUT
