#!/bin/bash
# 沈黙時間 5.58 ← 1秒以上の空白があるコメント集合における平均的な発話間隔
# label_prob = [0.82, 0.18]  # 全体のラベル割合分布
# TODO  1秒以上の空白があるコメント集合 を、スポッティングの対象とした方が簡単では？
## 0~1 秒の世界は難しすぎる

path="./data"

python src/main_v2.py --path $path --mean_silence_sec 5.58 --ignore_under_1sec

<< COMMENTOUT
test 集合での結果
diff_average: 2.9113
label_accuracy: 0.6925
confusion matrix: [[8262 1856]
 [1999  420]]
label_f1: 0.810
COMMENTOUT

echo "timing_algo: lognorm"
python src/main_v2.py --path $path --timing_algo lognorm --ignore_under_1sec --lognorm_params '{"shape": 0.8684, "loc": 0.0, "scale": 3.7154}'
# diff_average: 3.7224

echo "timing_algo: gamma"
python src/main_v2.py --path $path --timing_algo gamma --ignore_under_1sec --gamma_params '{"shape": 1.3725, "loc": 0.0, "scale": 4.0652}'
# diff_average: 3.9801

echo "timing_algo: expon"
python src/main_v2.py --path $path --timing_algo expon --ignore_under_1sec --expon_params '{"loc": 0.0, "scale": 5.5794}'
# diff_average: 4.1686