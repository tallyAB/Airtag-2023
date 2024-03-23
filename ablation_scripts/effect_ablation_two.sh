#!/bin/bash

# Cleanup working directories
directories=("../training_data/AblationTwo" "../embedding_data/AblationTwo" "../ground_truth/AblationTwo" "../models/AblationTwo" "../logs/AblationTwo")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done

# Create ablation set
nohup python create_ablation_sets.py --browser & wait

# Create new vocab from created data above
nohup python ../create_vocab.py  \
 -path /root/AirTag/training_data/AblationTwo/training_preprocessed_logs_S1-CVE-2015-5122_windows \
 -t_name /root/AirTag/training_data/AblationTwo/train_training_preprocessed_logs_S1-CVE-2015-5122_windows \
 -v_name /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S1-CVE-2015-5122_windows & 

nohup python ../create_vocab.py  \
 -path /root/AirTag/training_data/AblationTwo/training_preprocessed_logs_S2-CVE-2015-3105_windows \
 -t_name /root/AirTag/training_data/AblationTwo/train_training_preprocessed_logs_S2-CVE-2015-3105_windows \
 -v_name /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S2-CVE-2015-3105_windows &

nohup python ../create_vocab.py  \
 -path /root/AirTag/training_data/AblationTwo/training_preprocessed_logs_S3-CVE-2017-11882_windows \
 -t_name /root/AirTag/training_data/AblationTwo/train_training_preprocessed_logs_S3-CVE-2017-11882_windows \
 -v_name /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S3-CVE-2017-11882_windows &

nohup python ../create_vocab.py  \
 -path /root/AirTag/training_data/AblationTwo/training_preprocessed_logs_S4-CVE-2017-0199_windows_py \
 -t_name /root/AirTag/training_data/AblationTwo/train_training_preprocessed_logs_S4-CVE-2017-0199_windows_py \
 -v_name /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S4-CVE-2017-0199_windows_py &

wait

# Rename and remove extras
nohup mv /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S1-CVE-2015-5122_windows /root/AirTag/training_data/AblationTwo/vocab_atlas_single1_ablation_two.txt &
nohup mv /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S2-CVE-2015-3105_windows /root/AirTag/training_data/AblationTwo/vocab_atlas_single2_ablation_two.txt &
nohup mv /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S3-CVE-2017-11882_windows /root/AirTag/training_data/AblationTwo/vocab_atlas_single3_ablation_two.txt &
nohup mv /root/AirTag/training_data/AblationTwo/vocab_training_preprocessed_logs_S4-CVE-2017-0199_windows_py /root/AirTag/training_data/AblationTwo/vocab_atlas_single4_ablation_two.txt &

for file in /root/AirTag/training_data/AblationTwo/train_training*; do
  rm "$file" &
done

wait

nohup python ../create_pretraining_data.py  \
 --input_file=../training_data/AblationTwo/S1_train \
 --output_file=../training_data/AblationTwo/S1.tfrecord \
 --vocab_file=../training_data/AblationTwo/vocab_atlas_single1_ablation_two.txt \
 --do_lower_case=True \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

nohup python ../create_pretraining_data.py  \
  --input_file=../training_data/AblationTwo/S2_train \
  --output_file=../training_data/AblationTwo/S2.tfrecord \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single2_ablation_two.txt \
  --do_lower_case=True \
  --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

 nohup python ../create_pretraining_data.py  \
 --input_file=../training_data/AblationTwo/S3_train \
 --output_file=../training_data/AblationTwo/S3.tfrecord \
 --vocab_file=../training_data/AblationTwo/vocab_atlas_single3_ablation_two.txt \
 --do_lower_case=True \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

nohup python ../create_pretraining_data.py  \
 --input_file=../training_data/AblationTwo/S4_train \
 --output_file=../training_data/AblationTwo/S4.tfrecord \
 --vocab_file=../training_data/AblationTwo/vocab_atlas_single4_ablation_two.txt \
 --do_lower_case=True \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationTwo/S1.tfrecord \
  --output_dir=../models/AblationTwo/S1 \
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
 --learning_rate=2e-5 > ../logs/AblationTwo/trainS1.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationTwo/S2.tfrecord \
  --output_dir=../models/AblationTwo/S2 \
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
 --learning_rate=2e-5 > ../logs/AblationTwo/trainS2.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationTwo/S3.tfrecord \
  --output_dir=../models/AblationTwo/S3 \
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
 --learning_rate=2e-5 > ../logs/AblationTwo/trainS3.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationTwo/S4.tfrecord \
  --output_dir=../models/AblationTwo/S4 \
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
 --learning_rate=2e-5 > ../logs/AblationTwo/trainS4.log &

wait

# extract embeddings

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S1_benign \
  --output_file=../embedding_data/AblationTwo/S1_benign.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single1_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationTwo/S1/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S1.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S2_benign \
  --output_file=../embedding_data/AblationTwo/S2_benign.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single2_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationTwo/S2/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S2.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S3_benign \
  --output_file=../embedding_data/AblationTwo/S3_benign.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single3_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationTwo/S3/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S3.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S4_benign \
  --output_file=../embedding_data/AblationTwo/S4_benign.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single4_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationTwo/S4/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S4.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S1_test \
  --output_file=../embedding_data/AblationTwo/S1_test.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single1_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationTwo/S1/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S1_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S2_test \
  --output_file=../embedding_data/AblationTwo/S2_test.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single2_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
    --init_checkpoint=../models/AblationTwo/S2/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S2_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S3_test \
  --output_file=../embedding_data/AblationTwo/S3_test.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single3_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
   --init_checkpoint=../models/AblationTwo/S3/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S3_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationTwo/S4_test \
  --output_file=../embedding_data/AblationTwo/S4_test.json  \
  --vocab_file=../training_data/AblationTwo/vocab_atlas_single4_ablation_two.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationTwo/S4/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationTwo/extract_S4_test.log &

wait

nohup python -u evaluate_onesvm_Sdatasets_ablation_custom.py -flag 1 -nu 0.1 -gama 0.1 -gpu 0 -suffix AblationTwo > ../logs/AblationTwo/S1.log & wait
nohup python -u evaluate_onesvm_Sdatasets_ablation_custom.py -flag 2 -nu 0.1 -gama 0.15 -gpu 0 -suffix AblationTwo > ../logs/AblationTwo/S2.log & wait
nohup python -u evaluate_onesvm_Sdatasets_ablation_custom.py -flag 3 -nu 0.1 -gama 0.2 -gpu 0 -suffix AblationTwo > ../logs/AblationTwo/S3.log & wait
nohup python -u evaluate_onesvm_Sdatasets_ablation_custom.py -flag 4 -nu 0.1 -gama 0.15 -gpu 0 -suffix AblationTwo > ../logs/AblationTwo/S4.log & wait
