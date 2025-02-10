#!/bin/bash


$python train_sgan.py --n_train 30 --n_label 10 --add_supconloss True --temperature 0.3 --sup_wt 0.5 --df_aux_wt 0.5 --lr_g 0.0006 --lr_d 0.0001 --n_critic 2


