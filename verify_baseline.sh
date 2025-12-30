#!/bin/bash
python3 examples/basic_example.py > /dev/null 2>&1
if cmp -s basic_results/baseline_stats.json basic_results/baseline_stats_ref.json; then
    echo "True"
    exit 0
else
    echo "False"
    exit 1
fi
