# OStr-NAS

## Preparation
Follow https://github.com/D-X-Y/NAS-Bench-201 to install nasbench2:
`pip install nas-bench-201`, and download NAS-Bench-102-v1_0-e61699.pth.

## Search & Eval
Run the folowing command to perform search and evaluation simultaneously using one GPU:

`python train_nasbench.py --data $DATA_ROOT --set $DATASET --gpu $GPU`

The results will be saved and written to the log.
