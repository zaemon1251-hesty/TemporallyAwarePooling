#!/bin/bash

# 1秒未満を含めたデータでの 発話間隔の平均は 2.14 s

path="./data"

echo "System:"
echo "empirical, action_spotting, Separate, Addinfo force"
python src/main_v2.py --split test --path $path --seed 100 \
    --timing_algo empirical \
        --empirical_dist_csv  "/Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/silence_distribution.csv" \
        --mean_silence_sec 2.14 \
    --label_algo action_spotting \
        --action_window_size 15 \
        --action_rate_csv "/Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/Additional_Info_Ratios__Before_and_After.csv" \
        --default_rate 0.18 \
        --addinfo_force \
        --only_offplay
<< COMMENTOUT
diff_average: 40.450122607698894
label evaluation
confusion matrix: tn=36743 fp=19401 fn=7878 tp=4897
label_accuracy: 60.419%
label_precision: 20.154%
label_recall: 38.333%
label_f1: 26.418%
label w/ gold timing evaluation
confusion matrix: tn=37050 fp=19094 fn=7804 tp=4971
label_accuracy: 60.972%
label_precision: 20.657%
label_recall: 38.912%
label_f1: 26.987%
COMMENTOUT


echo "----------------------------------------"


echo "Baseline:"
echo "constant, constant,,"
python src/main_v2.py --split test --path $path --seed 100 \
    --timing_algo constant \
        --mean_silence_sec 2.14 \
    --label_algo constant
<< COMMENTOUT
diff_average: 19.864768786546527
label evaluation
confusion matrix: tn=46171 fp=9973 fn=10447 tp=2328
label_accuracy: 70.371%
label_precision: 18.925%
label_recall: 18.223%
label_f1: 18.568%
label w/ gold timing evaluation
confusion matrix: tn=46049 fp=10095 fn=10441 tp=2334
label_accuracy: 70.203%
label_precision: 18.779%
label_recall: 18.270%
label_f1: 18.521%
COMMENTOUT
