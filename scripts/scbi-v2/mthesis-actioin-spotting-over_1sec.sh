#!/bin/bash

# 1秒未満をのぞいたデータでの 発話間隔の平均は 5.58 s


path="./data"

echo "System:"
echo "empirical, action_spotting, Separate, Addinfo force"
python src/main_v2.py --split test --path $path --ignore_under_1sec --seed 100 \
    --timing_algo empirical \
        --empirical_dist_csv "/Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/silence_distribution_over_1sec.csv" \
    --label_algo action_spotting \
        --action_window_size 15 \
        --action_rate_csv "/Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/Additional_Info_Ratios__Before_and_After.csv" \
        --default_rate 0.18 \
        --addinfo_force \
        --only_offplay
<< COMMENTOUT
diff_average: 50.321448512403286
label evaluation
confusion matrix: tn=6651 fp=3467 fn=1482 tp=937
label_accuracy: 60.525%
label_precision: 21.276%
label_recall: 38.735%
label_f1: 27.466%
label w/ gold timing evaluation
confusion matrix: tn=6761 fp=3357 fn=1426 tp=993
label_accuracy: 61.849%
label_precision: 22.828%
label_recall: 41.050%
label_f1: 29.340%
COMMENTOUT


echo "----------------------------------------"


echo "Baseline:"
echo "constant, constant,,"
python src/main_v2.py --split test --path $path --ignore_under_1sec  --seed 100 \
    --timing_algo constant \
    --label_algo constant \
        --default_rate 0.18
<< COMMENTOUT
diff_average: 15.619845258036213
label evaluation
confusion matrix: tn=8330 fp=1788 fn=2006 tp=413
label_accuracy: 69.738%
label_precision: 18.764%
label_recall: 17.073%
label_f1: 17.879%
label w/ gold timing evaluation
confusion matrix: tn=8236 fp=1882 fn=1997 tp=422
label_accuracy: 69.060%
label_precision: 18.316%
label_recall: 17.445%
label_f1: 17.870%
COMMENTOUT