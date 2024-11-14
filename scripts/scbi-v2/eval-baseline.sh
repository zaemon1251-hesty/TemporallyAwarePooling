#!/bin/bash
# 沈黙時間 2.14 ← 平均的な発話間隔
# label_prob = [0.82, 0.18]  # 全体のラベル割合分布

path="./data"

echo "timing_algo: mean"
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

echo "timing_algo: lognorm"
python src/main_v2.py --path $path --timing_algo lognorm --lognorm_params '{"shape": 2.2926, "loc": 0.0, "scale": 0.2688}'
# diff_average: 4.7374

echo "timing_algo: gamma"
python src/main_v2.py --path $path --timing_algo gamma --gamma_params '{"shape": 0.3283, "loc": 0.0, "scale": 6.4844}'
# diff_average: 3.2179

echo "timing_algo: expon"
python src/main_v2.py --path $path --timing_algo expon --expon_params '{"loc": 0.0, "scale": 2.1289}'
# diff_average: 3.0265
