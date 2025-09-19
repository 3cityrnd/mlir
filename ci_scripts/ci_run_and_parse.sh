#!/bin/bash

if (($# < 2)); then
  echo "Usage: $0 artifact_dir commit1 commit2..." >&2
  exit 1
fi

start_time=$(date +%s)

ARTIFACT_DIR=$1
shift
COMMITS=("$@")

name_filters=()

for COMMIT in "${COMMITS[@]}"; do
  name_filters+=(-name "${COMMIT:0:7}_*" -o)
  ./ci_runner.sh $COMMIT $ARTIFACT_DIR
done

unset 'name_filters[${#name_filters[@]}-1]'

mapfile -t run_dirs < <(
    find $ARTIFACT_DIR \
        -maxdepth 1 \
        -type d \
        -newermt "@$start_time" \
        \( "${name_filters[@]}" \)
)

source ./venv/bin/activate
./ci_check_diffs.sh "${run_dirs[@]}" | python3 parse_to_pdf.py -o report_${start_time}.pdf