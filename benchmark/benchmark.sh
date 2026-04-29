#!/bin/bash

BASE_DIR=$(cd "$(dirname "$0")" && pwd)

INPUTS=("sample_2loop.txt" "abelian_higgs_mass.txt" "abelian_fermion_3loop.txt")
OUTPUT="$BASE_DIR/outputs"
TIME_FILE="$BASE_DIR/times.txt"

echo "File | Time (seconds)" > $TIME_FILE
echo "--------------------------" >> $TIME_FILE

for filename in "${INPUTS[@]}"; do
    input_path="$BASE_DIR/inputs/$filename"
    output_path="$OUTPUT/$filename"

    echo -e "\n\n\nProcessing $filename... \n\n\n"

    elapsed=$( (time -p sirena "$input_path" "$output_path" -p "params.txt" -vv) 2>&1 \
            | tee /dev/stderr \
            | awk '/real/ {print $2}' )

    echo "$filename | $elapsed s" >> "$TIME_FILE"
done