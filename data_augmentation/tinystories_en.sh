#!/bin/bash

# Get CPU cores
cores=$(nproc)
processes=0

# Languages
declare -A langs=(
  [en]="English"
)

# Loop languages
for lang in "${!langs[@]}"
do
  output_dir="./datasets/json_stories/${lang}"
  if [ ! -d "${output_dir}" ]; then
    mkdir -p "${output_dir}"
  fi

  # Loop datasets
  for i in {00..49}
  do
    input="./datasets/json_stories/archive/data${i}.json"

    # Check if already translated
    if [ ! -f "${output}" ]; then

      if [ "$processes" -ge 20 ]; then
        wait
        processes=0
        echo "processes cleared at $processes"
      fi

      # Construct command
      python3 aug_translation.py -i "${input}" -l "${lang}" > "data_${lang}_${i}.txt" &
      processes=$((processes+1))
      echo "number processes is $processes"

    fi
  done

  # Run parallel
done

wait
