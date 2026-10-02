#!/bin/bash
# Re-run kd_26s_v26_s0 after the AMP NaN fix (ultralytics_kd.py: KD terms in fp32, non-finite terms dropped).
# The first run diverged: NaN feature/logit KD batches from epoch 5, collapse at epochs 20-25 (val mask 0.161 -> 0).
# It is kept as kd_26s_v26_s0_diverged_amp. Waits for v26_chain.sh, then trains, averages and re-scores everything.
#   setsid nohup bash v2code/scripts/operations/v26_rerun_kd.sh > runs/logs/v26_rerun_kd.log 2>&1 < /dev/null & disown
W=/home/admin/distillation_workspace
C=$W/v2code
cd $W
export CPATH=$HOME/pyheaders/usr/include/python3.12:$HOME/pyheaders/usr/include
export PYTHONPATH=$C/src:$C/scripts
PY=$W/.venv/bin/python
P=$W/data/engine_bay_full_v10
S=$W/runs/segment
R=$W/runs/train_kd
T=engine_teacher_v26l
n=kd_26s_v26_s0
run() { local name=$1; shift; echo "$(date +%H:%M:%S) START $name"; "$@" > runs/logs/$name.log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) END $name rc=$rc"; return $rc; }

while pgrep -f "v26_chain.sh|v10_chain.sh|e1_kd_chain.sh" >/dev/null; do sleep 120; done
grep -q "_finite" $C/src/distillation/ultralytics_kd.py || { echo "KD NaN fix not synced"; exit 1; }

if [ -d $R/$n ] && [ ! -f $R/$n/FIXED_RUN ]; then
  mv $R/$n $R/${n}_diverged_amp && mv runs/logs/$n.log runs/logs/${n}_diverged_amp.log
fi
if [ ! -f $R/$n/weights/avg5.pt ]; then
  run $n $PY -u $C/scripts/training/train_kd.py --kd $C/configs/kd_hyperparams_p5.yaml --teacher $S/$T/weights/avg5.pt \
    --student yolo26s-seg.pt --data $P/data_vehicle.yaml --seed 0 --device 0 --project $R --name $n || exit 1
  touch $R/$n/FIXED_RUN
  run ${n}_avg $PY -u average_checkpoints.py --run $R/$n --top-k 5 --strip-kd --out $R/$n/weights/avg5.pt || exit 1
fi

M="--model teacher_v26l_avg5=$S/$T/weights/avg5.pt@640 --model teacher_v10_avg5=$S/engine_teacher_v10/weights/avg5.pt@640"
M="$M --model kd_n_v10_s0_avg5=$R/kd_n_v10_s0/weights/avg5.pt@640 --model kd_n_v10_s1_avg5=$R/kd_n_v10_s1/weights/avg5.pt@640"
for s in 0 1; do for kind in kd base; do
  f=$R/${kind}_26s_v26_s$s/weights/avg5.pt; [ -f $f ] && M="$M --model ${kind}_26s_v26_s${s}_avg5=$f@640"
done; done
run qa_v26b $PY -u qa_test.py $M --data $P/data_vehicle.yaml --split test --out qa_results_v26
echo "$(date +%H:%M:%S) V26 KD RERUN DONE (qa_results_v26/QA_REPORT.md)"
