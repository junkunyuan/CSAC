#!/usr/bin/env bash

# Set GPU ID.
gpu_id=0

# Run experiments on PACS dataset (ResNet-18).
python mains/exam_align.py --dataset pacs --gpu ${gpu_id} --test P --mode alpha > PACS_P.txt;

python mains/exam_align.py --dataset pacs --gpu ${gpu_id} --test A --mode alpha > PACS_A.txt;

python mains/exam_align.py --dataset pacs --gpu ${gpu_id} --test C --mode alpha > PACS_C.txt;

python mains/exam_align.py --dataset pacs --gpu ${gpu_id} --test S --mode alpha > PACS_S.txt;

# Run experiments on VLCS dataset (ResNet-18).
python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test P --mode alpha > VLCS_P.txt;

python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test L --mode alpha > VLCS_L.txt;

python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test C --mode alpha > VLCS_C.txt;

python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test S --mode alpha > VLCS_S.txt;

# Run experiments on VLCS dataset (AlextNet).
python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test P --mode alpha --net alexnet > VLCS_alex_P.txt;

python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test L --mode alpha --net alexnet > VLCS_alex_L.txt;

python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test C --mode alpha --net alexnet > VLCS_alex_C.txt;

python mains/exam_align.py --dataset vlcs --gpu ${gpu_id} --test S --mode alpha --net alexnet > VLCS_alex_S.txt;
