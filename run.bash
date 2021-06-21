#! /usr/bin/env bash
set -eou pipefail

results_base_path="$CSC500_ROOT_PATH/csc500-past-runs/dann-cida/"


for batch_size in 64 128; do
for epochs in 25; do
for learning_rate in 0.0001; do
for src_distance in "2.8.14.20.26"; do
for target_distance in 32; do
for alpha in 0.001 0.1 0.01; do
for num_additional_extractor_fc_layers in 0 1 2 3; do
    experiment_name=dann.cida_src.distance-${src_distance}_targetDistance-${target_distance}_alpha-${alpha}_learningRate-${learning_rate}_batch-${batch_size}_epochs-${epochs}_numAdditionalExtractorLayers-${num_additional_extractor_fc_layers}
    echo "Begin $experiment_name" | tee logs
    rm -rf *png logs experiment_name
    echo $experiment_name > experiment_name
    cat << EOF | python3 ./main.py 2>&1 | tee --append logs
    {
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
