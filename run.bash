#! /usr/bin/env bash
set -eou pipefail


results_base_path="$CSC500_ROOT_PATH/csc500-past-runs/chapter2/jsonTesting"

patience=100

ALL_SERIAL_NUMBERS='["3123D52","3123D65","3123D79","3123D80","3123D54","3123D70","3123D7B","3123D89","3123D58","3123D76","3123D7D","3123EFE","3123D64","3123D78","3123D7E","3124E4A"]'

for seed in 6031 1710 4253 8134 5133 9001 1337; do
for batch_size in 256; do
# for epochs in 1000; do
for epochs in 3; do
for learning_rate in 0.0001; do
# for source_distance in "[2]" "[8]"; do
for source_distance in "[14]"; do
for target_distance in "[14]"; do
for alpha in 0.001; do
for num_additional_extractor_fc_layers in 1; do
for window_stride in 50; do
for window_length in 256; do
for num_examples_per_device in 100; do
for desired_runs in "[1]"; do
for desired_serial_numbers in "$ALL_SERIAL_NUMBERS"; do
# for seed in 8646; do
    experiment_name=name:cnnOnly-Dummy-nllLoss
    experiment_name=${experiment_name}_tipoff:yes
    experiment_name=${experiment_name}_seed:${seed}
    experiment_name=${experiment_name}_srcDistance:${source_distance}
    experiment_name=${experiment_name}_targetDistance:${target_distance}
    # experiment_name=${experiment_name}_alpha:${alpha}
    experiment_name=${experiment_name}_learningRate:${learning_rate}
    experiment_name=${experiment_name}_batchSize:${batch_size}
    experiment_name=${experiment_name}_epochs:${epochs}
    # experiment_name=${experiment_name}_numAdditionalExtractorLayers:${num_additional_extractor_fc_layers}
    experiment_name=${experiment_name}_patience:${patience}
    experiment_name=${experiment_name}_examplesPerDevice:${num_examples_per_device}
    experiment_name=${experiment_name}_windowStride:${window_stride}
    experiment_name=${experiment_name}_windowLength:${window_length}
    experiment_name=${experiment_name}_runs:${desired_runs}

    if [[ "$desired_serial_numbers" == "$ALL_SERIAL_NUMBERS" ]]; then
        experiment_name=${experiment_name}_serials:ALL
    else
        experiment_name=${experiment_name}_serials:${desired_serial_numbers}
    fi

    rm -rf results
    mkdir results
    echo "Begin $experiment_name" | tee results/logs

    cat << EOF | python3 ./main.py 2>&1 | tee --append results/logs
    {
        "seed": $seed,
        "patience": $patience,
        "experiment_name": "$experiment_name",
        "lr": $learning_rate,
        "n_epoch": $epochs,
        "batch_size": $batch_size,
        "source_distance": $source_distance,
        "target_distance": $target_distance,
        "alpha": $alpha,
        "num_additional_extractor_fc_layers": $num_additional_extractor_fc_layers,
        "desired_serial_numbers": $desired_serial_numbers,
        "num_examples_per_device": $num_examples_per_device,
        "window_stride": $window_stride,
        "window_length": $window_length,
        "desired_runs": $desired_runs
    }
EOF

    mkdir -p $results_base_path

    cp -R --no-clobber . $results_base_path/$experiment_name
    rm $results_base_path/$experiment_name/.gitignore
    rm -rf $results_base_path/$experiment_name/__pycache__

done
done
done
done
done
done
done
done
done
done
done
done
done
