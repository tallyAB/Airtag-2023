#!/bin/bash

# Cleanup working directories
directories=("../training_data/AblationOne" "../embedding_data/AblationOne" "../ground_truth/AblationOne" "../models/AblationOne" "../logs/AblationOne")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done

# Create ablation set
nohup python create_ablation_sets.py --dns --browser & wait

# Create new vocab from created data above
nohup python ../create_vocab_atlas.py  \
 -path /root/AirTag/training_data/AblationOne/training_preprocessed_logs_S1-CVE-2015-5122_windows \
 -t_name train_training_preprocessed_logs_S1-CVE-2015-5122_windows \
 -v_name vocab_training_preprocessed_logs_S1-CVE-2015-5122_windows & 

nohup python ../create_vocab_atlas.py  \
 -path /root/AirTag/training_data/AblationOne/training_preprocessed_logs_S2-CVE-2015-3105_windows \
 -t_name train_training_preprocessed_logs_S2-CVE-2015-3105_windows \
 -v_name vocab_training_preprocessed_logs_S2-CVE-2015-3105_windows &

nohup python ../create_vocab_atlas.py  \
 -path /root/AirTag/training_data/AblationOne/training_preprocessed_logs_S3-CVE-2017-11882_windows \
 -t_name train_training_preprocessed_logs_S3-CVE-2017-11882_windows \
 -v_name vocab_training_preprocessed_logs_S3-CVE-2017-11882_windows &

nohup python ../create_vocab_atlas.py  \
 -path /root/AirTag/training_data/AblationOne/training_preprocessed_logs_S4-CVE-2017-0199_windows_py \
 -t_name train_training_preprocessed_logs_S4-CVE-2017-0199_windows_py \
 -v_name vocab_training_preprocessed_logs_S4-CVE-2017-0199_windows_py &

wait

# Rename and remove extras
nohup mv /root/AirTag/training_data/AblationOne/vocab_training_preprocessed_logs_S1-CVE-2015-5122_windows /root/AirTag/training_data/AblationOne/vocab_atlas_single1_ablation_one.txt &
nohup mv /root/AirTag/training_data/AblationOne/vocab_training_preprocessed_logs_S2-CVE-2015-3105_windows /root/AirTag/training_data/AblationOne/vocab_atlas_single2_ablation_one.txt &
nohup mv /root/AirTag/training_data/AblationOne/vocab_training_preprocessed_logs_S3-CVE-2017-11882_windows /root/AirTag/training_data/AblationOne/vocab_atlas_single3_ablation_one.txt &
nohup mv /root/AirTag/training_data/AblationOne/vocab_training_preprocessed_logs_S4-CVE-2017-0199_windows_py /root/AirTag/training_data/AblationOne/vocab_atlas_single4_ablation_one.txt &

for file in /root/AirTag/training_data/AblationOne/train_training*; do
  rm "$file" &
done

wait

nohup python ../create_pretraining_data.py  \
 --input_file=../training_data/AblationOne/S1_train \
 --output_file=../training_data/AblationOne/S1.tfrecord \
 --vocab_file=../training_data/AblationOne/vocab_atlas_single1_ablation_one.txt \
 --do_lower_case=True \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

nohup python ../create_pretraining_data.py  \
  --input_file=../training_data/AblationOne/S2_train \
  --output_file=../training_data/AblationOne/S2.tfrecord \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single2_ablation_one.txt \
  --do_lower_case=True \
  --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

 nohup python ../create_pretraining_data.py  \
 --input_file=../training_data/AblationOne/S3_train \
 --output_file=../training_data/AblationOne/S3.tfrecord \
 --vocab_file=../training_data/AblationOne/vocab_atlas_single3_ablation_one.txt \
 --do_lower_case=True \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

nohup python ../create_pretraining_data.py  \
 --input_file=../training_data/AblationOne/S4_train \
 --output_file=../training_data/AblationOne/S4.tfrecord \
 --vocab_file=../training_data/AblationOne/vocab_atlas_single4_ablation_one.txt \
 --do_lower_case=True \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationOne/S1.tfrecord \
  --output_dir=../models/AblationOne/S1 \
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
 --learning_rate=2e-5 > ../logs/AblationOne/trainS1.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationOne/S2.tfrecord \
  --output_dir=../models/AblationOne/S2 \
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
 --learning_rate=2e-5 > ../logs/AblationOne/trainS2.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationOne/S3.tfrecord \
  --output_dir=../models/AblationOne/S3 \
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
 --learning_rate=2e-5 > ../logs/AblationOne/trainS3.log &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/AblationOne/S4.tfrecord \
  --output_dir=../models/AblationOne/S4 \
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
 --learning_rate=2e-5 > ../logs/AblationOne/trainS4.log &

wait

# extract embeddings

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S1_benign \
  --output_file=../embedding_data/AblationOne/S1_benign.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single1_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationOne/S1/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S1.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S2_benign \
  --output_file=../embedding_data/AblationOne/S2_benign.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single2_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationOne/S2/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S2.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S3_benign \
  --output_file=../embedding_data/AblationOne/S3_benign.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single3_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationOne/S3/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S3.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S4_benign \
  --output_file=../embedding_data/AblationOne/S4_benign.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single4_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationOne/S4/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S4.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S1_test \
  --output_file=../embedding_data/AblationOne/S1_test.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single1_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationOne/S1/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S1_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S2_test \
  --output_file=../embedding_data/AblationOne/S2_test.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single2_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
    --init_checkpoint=../models/AblationOne/S2/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S2_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S3_test \
  --output_file=../embedding_data/AblationOne/S3_test.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single3_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
   --init_checkpoint=../models/AblationOne/S3/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S3_test.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/AblationOne/S4_test \
  --output_file=../embedding_data/AblationOne/S4_test.json  \
  --vocab_file=../training_data/AblationOne/vocab_atlas_single4_ablation_one.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/AblationOne/S4/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/AblationOne/extract_S4_test.log &

wait

nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 1 -nu 0.1 -gama 0.1 -gpu 0 -suffix AblationOne > ../logs/AblationOne/S1.log & wait
nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 2 -nu 0.1 -gama 0.15 -gpu 0 -suffix AblationOne > ../logs/AblationOne/S2.log & wait
nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 3 -nu 0.1 -gama 0.2 -gpu 0 -suffix AblationOne > ../logs/AblationOne/S3.log & wait
nohup python -u evaluate_onesvm_Sdatasets_custom.py -flag 4 -nu 0.1 -gama 0.15 -gpu 0 -suffix AblationOne > ../logs/AblationOne/S4.log & wait
