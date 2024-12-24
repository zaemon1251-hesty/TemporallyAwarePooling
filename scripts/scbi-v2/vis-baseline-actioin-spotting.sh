#!/bin/bash


path="./data"


echo "label_algo: action_spotting,Separate former and latter,Addinfo force"
python src/main_v2.py --split valid --path $path --ignore_under_1sec  \
    --seed 100 \
    --label_algo action_spotting --action_window_size 15 \
    --timing_algo expon --expon_params '{"loc": 1.0, "scale": 4.5825}' \
    --action_rate_csv "/Users/heste/workspace/soccernet/sn-caption/Benchmarks/TemporallyAwarePooling/data/Additional_Info_Ratios__Before_and_After.csv" \


<< COMMENTOUT
test 集合での結果
--action_window_size 10
label_accuracy: 0.701
confusion matrix: [[8309 1809]
 [1928  491]]
label_f1: 0.816
COMMENTOUT

# echo "timing_algo: lognorm"
# python src/main_v2.py --path $path --label_algo action_spotting --action_window_size 15 --seed 0 --timing_algo lognorm --ignore_under_1sec --lognorm_params '{"shape": 1.4902, "loc": 1.0, "scale": 1.9815}'
# # diff_average: 5.4240

# echo "timing_algo: gamma"
# python src/main_v2.py --path $path --label_algo action_spotting --action_window_size 15 --seed 0 --timing_algo gamma --ignore_under_1sec --gamma_params '{"shape": 0.7183, "loc": 1.0, "scale": 6.3800}'
# # diff_average: 3.8224

# echo "timing_algo: expon"
# python src/main_v2.py --path $path --label_algo action_spotting --action_window_size 15 --seed 0 --timing_algo expon --ignore_under_1sec --expon_params '{"loc": 1.0, "scale": 4.5825}'
# # diff_average: 3.5467