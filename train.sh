#!/bin/bash

echo "Run pre_syn_word"
python train.py \
  --fold 0 \
  --config "config/pretrained_syn_word.gin" \
  --model_type "smt" \
  --batch_size 64 \
  --accumulate_grad_batches 1 \
  --lr 5e-4 \
  --epochs 100 \
  --patience 10

echo "Run pre_syn_character"
python train.py \
  --fold 0 \
  --config "config/pretrained_syn_character.gin" \
  --model_type "smt" \
  --batch_size 64 \
  --accumulate_grad_batches 1 \
  --lr 1e-5 \
  --epochs 100 \
  --patience 10

echo "Run pre_character"
python train.py \
  --fold 0 \
  --config "config/pretrain_character.gin" \
  --model_type "smt" \
  --batch_size 64 \
  --accumulate_grad_batches 1 \
  --lr 1e-5 \
  --epochs 100 \
  --patience 10
 
 echo "Run syn_medium"
python train.py \
  --fold 0 \
  --config "config/pretrain_medium.gin" \
  --model_type "smt" \
  --batch_size 64 \
  --accumulate_grad_batches 1 \
  --lr 5e-4 \
  --epochs 100 \
  --patience 10

echo "Run pre_syn_word"
python train.py \
  --fold 0 \
  --config "config/pretrain_syn_word.gin" \
  --model_type "smt" \
  --batch_size 64 \
  --accumulate_grad_batches 1 \
  --lr 1e-5 \
  --epochs 100 \
  --patience 10


