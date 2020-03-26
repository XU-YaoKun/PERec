!#/usr/bin/env bash
CFG="configs/DNS.yaml"

for i in 1 2 3
do
    CUDA_VISIBLE_DEVICES=0 python main.py --cfg $CFG >> DNS.log
done

