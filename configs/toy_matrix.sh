#!/bin/bash
# The CPU toy matrix: methods x rewards on one environment. ~1 s/step at batch 8 on a laptop for the 331M model.
#   bash configs/toy_matrix.sh synth            # or: bash configs/toy_matrix.sh csv:data/electricity.csv
set -e
BANK=${1:-synth}; NAME=$(basename "${BANK#csv:}" .csv); STEPS=${STEPS:-300}; OUT=runs/$NAME; mkdir -p $OUT
run() { echo "[$(date +%H:%M)] $1"; python3 scripts/run_matrix.py --bank $BANK --steps $STEPS --out $OUT/$1 "${@:2}"; }
run native               --algo none
run sft_pinball          --algo sft        --reward pinball                          # M1  SFT control
run sft_pinball_seed1    --algo sft        --reward pinball --seed 1                 #     noise bound
run grpo_mse             --algo grpo       --reward mse                              # M2  TS-GRPO
run grpo_mse_anchor      --algo grpo       --reward mse --anchor 0.1                 # M3  GTN-R-like
run grpo_mse_perstep     --algo grpo       --reward mse --per_step                   # M4  TimeRFT-like credit
run pref_mse             --algo preference --reward mse                              # M5  TPO-like
run ppo_composite        --algo ppo        --reward composite                        # M6  PPO, our reward
run ppo_mse              --algo ppo        --reward mse
run grpo_crps            --algo grpo       --reward crps                             #     reward ablation
run grpo_skill           --algo grpo       --reward skill
run grpo_composite       --algo grpo       --reward composite                        # M7  ours
run grpo_composite_head  --algo grpo       --reward composite --head_adapter         # M7 + our policy class
run context_ipo          --algo context    --reward composite                        # M8  context policy (frozen forecaster)
python3 scripts/make_report.py $OUT > $OUT/report.md; cat $OUT/report.md
