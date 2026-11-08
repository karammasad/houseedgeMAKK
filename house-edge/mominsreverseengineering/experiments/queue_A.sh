#!/bin/zsh
# Runs candidate A (gate family capped at lag 8, the organizers' stated maximum) once the baseline score (pid in $1) has finished.
# Tuning half scored first via run.py (also logs audit/full to experiments/ledger.csv); audit is read from the same row.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 5; done
python3 run.py table
python3 run.py score --name "A gate max_lag 8 (lags<=8)" --kwargs '{"gate":{"max_lag":8,"and_pairs":true,"prior_flips":true,"prune":0.01,"floor":0.05,"cell_prior":"learn","cell_group":"type","p_change_short":"learn","sizes":"npmle","size_split":true}}'
python3 run.py table
