#!/bin/bash
cd /home/claude/hbn; export PYTHONPATH=.
S="ponce_pr seribu_id langebaan_za kerala_in"
while pgrep -f "run_study cv --sites" >/dev/null; do sleep 20; done
python3 -m hbn2.run_study loso --sites $S --target kerala_in --members 1 --out runs_v2/loso >> runs_v2/loso.out 2>&1
python3 -m hbn2.run_study loso --sites $S --target kerala_in --adapt --members 1 --out runs_v2/loso >> runs_v2/loso.out 2>&1
for c in cnn_only no_physics_loss; do
  python3 -m hbn2.run_study cv --sites $S --config $c --members 1 --out runs_v2/abl_$c >> runs_v2/abl.out 2>&1
done
echo ALLDONE >> runs_v2/queue.log
