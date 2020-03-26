!#/usr/bin/env bash
CFG="configs/NMRN.yaml"
NUM_BATCH=256

# when NUM_BATCH=256, 6GiB is required to run the script.
# If NUM_BATCH is smaller, much more cuda memory is required. 
for i in 1 2 3
do
    CUDA_VISIBLE_DEVICES=0 python main.py --cfg $CFG TEST.NUM_BATCH $NUM_BATCH >> NMRN.log
done

