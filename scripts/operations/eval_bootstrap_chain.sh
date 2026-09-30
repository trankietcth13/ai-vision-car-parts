#!/bin/bash
# Paired image-bootstrap comparisons (scripts/evaluation/paired_bootstrap.py) for every candidate vs the p5 references.
# Replaces "2-seed mean >= reference + 0.01" by a difference with a 95% interval over resampled test images.
#   old test  (125 imgs, data/engine_bay_train_v8)  : v9 teacher/KD, E1 KD  vs p5_reg / kd_n_p5t
#   v2tax test (data/engine_bay_full_v2, 27 classes) : v2tax teacher vs p5_reg on the SHARED images and class names
#   test v2   (patched labels, data/engine_bay_full_v10): v10 teacher/KD vs p5_reg / kd_n_p5t
# Runs after every training chain (one GPU job at a time); each collect is one validation pass (~20 s per model).
#   setsid nohup bash v2code/scripts/operations/eval_bootstrap_chain.sh > runs/logs/eval_bootstrap_chain.log 2>&1 < /dev/null & disown
W=/home/admin/distillation_workspace
C=$W/v2code
cd $W
export PYTHONPATH=$C/src:$C/scripts
PY=$W/.venv/bin/python
PB=$C/scripts/evaluation/paired_bootstrap.py
S=$W/runs/segment
R=$W/runs/train_kd
E=$W/eval_stats
O=$W/eval_reports
mkdir -p $E $O
run() { local name=$1; shift; echo "$(date +%H:%M:%S) START $name"; "$@" > runs/logs/$name.log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) END $name rc=$rc"; return $rc; }

while pgrep -f "phase3_chain.sh|v2_teacher_chain.sh|e1_kd_chain.sh|v10_chain.sh" >/dev/null; do sleep 300; done

m() { [ -f "$2" ] && echo "--model $1=$2@640"; }  # only models whose weights exist
P5="$(m p5_reg $S/p5_reg/weights/avg5.pt) $(m kd_n_p5t_s0 $R/kd_n_p5t_s0/weights/avg5.pt) $(m kd_n_p5t_s1 $R/kd_n_p5t_s1/weights/avg5.pt)"

# 1) old test
run pb_collect_old $PY -u $PB collect --data $W/data/engine_bay_train_v8/data_engine_bay_train.yaml --out $E/test_old $P5 \
  $(m teacher_v9 $S/engine_teacher_v9/weights/avg5.pt) $(m kd_n_v9_s0 $R/kd_n_v9_s0/weights/avg5.pt) $(m kd_n_v9_s1 $R/kd_n_v9_s1/weights/avg5.pt) \
  $(m kd_n_e1_s0 $R/kd_n_e1_s0/weights/avg5.pt) $(m kd_n_e1_s1 $R/kd_n_e1_s1/weights/avg5.pt)
run pb_compare_old $PY -u $PB compare --stats $E/test_old \
  --group p5_kd=kd_n_p5t_s0,kd_n_p5t_s1 --group v9_kd=kd_n_v9_s0,kd_n_v9_s1 --group e1_kd=kd_n_e1_s0,kd_n_e1_s1 \
  --pair teacher_v9:p5_reg --pair v9_kd:p5_kd --pair e1_kd:p5_kd --out $O/bootstrap_test_old.md

# 2) v2 taxonomy teacher vs p5 on shared images / class names (the report states whether the labels are identical)
run pb_collect_v2tax $PY -u $PB collect --data $W/data/engine_bay_full_v2/data_vehicle.yaml --out $E/test_v2tax \
  $(m teacher_v2tax $S/engine_teacher_v2tax/weights/avg5.pt)
run pb_compare_v2tax $PY -u $PB compare --file $E/test_v2tax/teacher_v2tax.npz --file $E/test_old/p5_reg.npz \
  --pair teacher_v2tax:p5_reg --out $O/bootstrap_v2tax_vs_p5.md

# 3) test v2 (patched labels): v10 vs p5 references
if [ -f $W/data/engine_bay_full_v10/data_vehicle.yaml ]; then
  run pb_collect_v10 $PY -u $PB collect --data $W/data/engine_bay_full_v10/data_vehicle.yaml --out $E/test_v10 $P5 \
    $(m teacher_v10 $S/engine_teacher_v10/weights/avg5.pt) $(m kd_n_v10_s0 $R/kd_n_v10_s0/weights/avg5.pt) $(m kd_n_v10_s1 $R/kd_n_v10_s1/weights/avg5.pt)
  run pb_compare_v10 $PY -u $PB compare --stats $E/test_v10 \
    --group p5_kd=kd_n_p5t_s0,kd_n_p5t_s1 --group v10_kd=kd_n_v10_s0,kd_n_v10_s1 \
    --pair teacher_v10:p5_reg --pair v10_kd:p5_kd --out $O/bootstrap_test_v10.md
fi
echo "$(date +%H:%M:%S) EVAL BOOTSTRAP DONE ($O)"
