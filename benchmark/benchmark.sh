#!/bin/bash

BASE_DIR=$(cd "$(dirname "$0")" && pwd)

INPUTS=("sample_2loop.txt" "abelian_higgs_mass.txt" "abelian_fermion_3loop.txt")
OUTPUT="$BASE_DIR/outputs"
TIME_FILE="$BASE_DIR/times.txt"

PARAMS_FILE1="params_sample.txt"
PARAMS_FILE2="params_abelian.txt"

echo "File | Time (seconds)" > $TIME_FILE
echo "--------------------------" >> $TIME_FILE

for filename in "${INPUTS[@]}"; do
    input_path="$BASE_DIR/inputs/$filename"
    output_path="$OUTPUT/$filename"

    if [[ "$filename" == "sample_2loop.txt" ]]; then
        PARAMS="$PARAMS_FILE1"
    else
        PARAMS="$PARAMS_FILE2"
    fi

    echo -e "\n\n\nProcessing $filename with $PARAMS... \n\n\n"

    elapsed=$( (time -p sirena "$input_path" "$output_path" -p "$PARAMS" -vv) 2>&1 \
            | tee /dev/stderr \
            | awk '/real/ {print $2}' )

    echo "$filename | $elapsed s" >> "$TIME_FILE"
done