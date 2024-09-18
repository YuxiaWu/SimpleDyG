#!/bin/bash
for Timestamp in 12
do
    for batch_size in 32 
    do
        for n_layer in 6
        do
            for n_head in 8
            do
                for n_embed in 768
                do
                    for lr in 0.0001 
                    do
                        for seed in 0 1 2 3 4 5 6 7 8 9
                            do
                            export TRAIN_FILE="./resources/UCI_13/$Timestamp/train.link_prediction"
                            export TEST_FILE="./resources/UCI_13/$Timestamp/test.link_prediction"
                            export TEST_GT_FILE="./resources/UCI_13/$Timestamp/test_gt.link_prediction"
                            export VAL_FILE="./resources/UCI_13/$Timestamp/val.link_prediction"    
                            export VAL_GT_FILE="./resources/UCI_13/$Timestamp/val_gt.link_prediction"    
                            export output="output/UCI_13/$Timestamp/{$n_layer}_{$n_head}_{$n_embed}_{$batch_size}_{$lr}_{$seed}/gpt2"

                            CUDA_VISIBLE_DEVICES=2 python main.py \
                                --dataset 'UCI_13' \
                                --output_dir=$output \
                                --model_type 'gpt2' \
                                --model_name_or_path 'gpt2' \
                                --train_data_file=$TRAIN_FILE \
                                --do_train \
                                --eval_data_file=$VAL_FILE \
                                --eval_data_gt_file=$VAL_GT_FILE \
                                --test_data_file=$TEST_FILE \
                                --test_data_gt_file=$TEST_GT_FILE \
                                --save_steps 250 \
                                --logging_steps 500 \
                                --per_gpu_train_batch_size=$batch_size \
                                --num_train_epochs 100 \
                                --block_size 512 \
                                --eval_all_checkpoints \
                                --timestamp $Timestamp \
                                --patience 10 \
                                --n_layer=$n_layer \
                                --n_head=$n_head \
                                --n_embed=$n_embed \
                                --learning_rate=$lr \
                                --seed=$seed \
                                --run_seed                                               
                            done
                        done
                    done
                done
            done
        done
    done
done

