#!/bin/bash

# Cleanup working directories
# directories=("../training_data/Original" "../embedding_data/Original" "../ground_truth/Original" "../models/Original" "../logs/Original")
# for dir in "${directories[@]}"; do
#     if [ -d "$dir" ]; then
#         rm -rf "$dir"
#     fi
# done
# for dir in "${directories[@]}"; do
#     mkdir -p "$dir"
# done

# nohup cp ../training_data/S1_train \
#  ../training_data/S1_benign \
#  ../training_data/S1_test \
#  ../training_data/S2_train \
#  ../training_data/S2_benign \
#  ../training_data/S2_test \
#  ../training_data/S3_train \
#  ../training_data/S3_benign \
#  ../training_data/S3_test \
#  ../training_data/S4_train \
#  ../training_data/S4_benign \
#  ../training_data/S4_test \
#  ../training_data/training_preprocessed_logs_S1-CVE-2015-5122_windows \
#  ../training_data/training_preprocessed_logs_S2-CVE-2015-3105_windows \
#  ../training_data/training_preprocessed_logs_S3-CVE-2017-11882_windows \
#  ../training_data/training_preprocessed_logs_S4-CVE-2017-0199_windows_py \
#  ../training_data/vocab_atlas_single1.txt \
#  ../training_data/vocab_atlas_single2.txt \
#  ../training_data/vocab_atlas_single3.txt \
#  ../training_data/vocab_atlas_single4.txt \
#  ../training_data/Original &

# nohup cp ../ground_truth/S1_number_.npy \
#  ../ground_truth/S2_number_.npy \
#  ../ground_truth/S3_number_.npy \
#  ../ground_truth/S4_number_.npy \
#  ../ground_truth/Original &

# wait

# nohup python ../create_pretraining_data.py  \
#  --input_file=../training_data/Original/S1_train \
#  --output_file=../training_data/Original/S1.tfrecord \
#  --vocab_file=../training_data/Original/vocab_atlas_single1.txt \
#  --do_lower_case=True \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &

# nohup python ../create_pretraining_data.py  \
#   --input_file=../training_data/Original/S2_train \
#   --output_file=../training_data/Original/S2.tfrecord \
#   --vocab_file=../training_data/Original/vocab_atlas_single2.txt \
#   --do_lower_case=True \
#   --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &

#  nohup python ../create_pretraining_data.py  \
#  --input_file=../training_data/Original/S3_train \
#  --output_file=../training_data/Original/S3.tfrecord \
#  --vocab_file=../training_data/Original/vocab_atlas_single3.txt \
#  --do_lower_case=True \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &

# nohup python ../create_pretraining_data.py  \
#  --input_file=../training_data/Original/S4_train \
#  --output_file=../training_data/Original/S4.tfrecord \
#  --vocab_file=../training_data/Original/vocab_atlas_single4.txt \
#  --do_lower_case=True \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &

# wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/Original/S1.tfrecord \
  --output_dir=../models/Original/S1 \
  --do_train=True \
  --do_eval=True  \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../uncased_L-6_H-128_A-2/bert_model.ckpt \
  --train_batch_size=4  \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --num_train_steps=10000  \
 --num_warmup_steps=10  \
 --gpu=0  \
 --learning_rate=2e-5 > ../logs/Original/trainS1.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/Original/S2.tfrecord \
  --output_dir=../models/Original/S2 \
  --do_train=True \
  --do_eval=True  \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../uncased_L-6_H-128_A-2/bert_model.ckpt \
  --train_batch_size=4  \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --num_train_steps=10000  \
 --num_warmup_steps=10  \
 --gpu=0  \
 --learning_rate=2e-5 > ../logs/Original/trainS2.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/Original/S3.tfrecord \
  --output_dir=../models/Original/S3 \
  --do_train=True \
  --do_eval=True  \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../uncased_L-6_H-128_A-2/bert_model.ckpt \
  --train_batch_size=4  \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --num_train_steps=10000  \
 --num_warmup_steps=10  \
 --gpu=0  \
 --learning_rate=2e-5 > ../logs/Original/trainS3.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/Original/S4.tfrecord \
  --output_dir=../models/Original/S4 \
  --do_train=True \
  --do_eval=True  \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../uncased_L-6_H-128_A-2/bert_model.ckpt \
  --train_batch_size=4  \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --num_train_steps=10000  \
 --num_warmup_steps=10  \
 --gpu=0  \
 --learning_rate=2e-5 > ../logs/Original/trainS4.log &

wait

# extract embeddings

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S1_benign \
  --output_file=../embedding_data/Original/S1_benign.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single1.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/Original/S1/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S1.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S2_benign \
  --output_file=../embedding_data/Original/S2_benign.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single2.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/Original/S2/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S2.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S3_benign \
  --output_file=../embedding_data/Original/S3_benign.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single3.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/Original/S3/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S3.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S4_benign \
  --output_file=../embedding_data/Original/S4_benign.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single4.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/Original/S4/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S4.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S1_test \
  --output_file=../embedding_data/Original/S1_test.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single1.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/Original/S1/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S1_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S2_test \
  --output_file=../embedding_data/Original/S2_test.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single2.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
    --init_checkpoint=../models/Original/S2/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S2_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S3_test \
  --output_file=../embedding_data/Original/S3_test.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single3.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
   --init_checkpoint=../models/Original/S3/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S3_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/Original/S4_test \
  --output_file=../embedding_data/Original/S4_test.json  \
  --vocab_file=../training_data/Original/vocab_atlas_single4.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/Original/S4/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/Original/extract_S4_test.log &

wait

nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 1 -nu 0.1 -gama 0.1 -gpu 0 -suffix Original > ../logs/Original/S1.log & wait
nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 2 -nu 0.1 -gama 0.15 -gpu 0 -suffix Original > ../logs/Original/S2.log & wait
nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 3 -nu 0.1 -gama 0.2 -gpu 0 -suffix Original > ../logs/Original/S3.log & wait
nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 4 -nu 0.1 -gama 0.15 -gpu 0 -suffix Original > ../logs/Original/S4.log & wait
