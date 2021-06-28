#! /usr/bin/env bash
set -eou pipefail

results_base_path="$CSC500_ROOT_PATH/csc500-past-runs/dann-cida-yolo/"

patience=10

for batch_size in 64; do
for epochs in 15; do
for learning_rate in 0.0001; do
for src_distance in "2.8.14.20.26"; do
for target_distance in 32; do
for alpha in 0.001; do
for num_additional_extractor_fc_layers in 1; do
for seed in 8646 25792 15474 5133 30452 17665 27354 17752; do
    experiment_name=dann.cida.yolo_tipoff-no_seed-${seed}_src.distance-${src_distance}_targetDistance-${target_distance}_alpha-${alpha}_learningRate-${learning_rate}_batch-${batch_size}_epochs-${epochs}_numAdditionalExtractorLayers-${num_additional_extractor_fc_layers}
    echo "Begin $experiment_name" | tee logs
    rm -rf *png logs experiment_name
    echo $experiment_name > experiment_name
    cat << EOF | python3 ./main.py 2>&1 | tee --append logs
    {
        "seed": $seed,
        "patience": $patience,
        "experiment_name": "$experiment_name",
        "lr": $learning_rate,
        "n_epoch": $epochs,
        "batch_size": $batch_size,
        "source_distance": "$src_distance",
        "target_distance": "$target_distance",
        "alpha": $alpha,
        "num_additional_extractor_fc_layers": $num_additional_extractor_fc_layers
    }
EOF

    cp -R . $results_base_path/$experiment_name
    rm $results_base_path/$experiment_name/.gitignore

done
done
done
done
done
done
done
done
