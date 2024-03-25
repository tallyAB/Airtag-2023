#!/bin/bash

# Cleanup working directories
directories=("../training_data/SIGL" "../embedding_data/SIGL" "../ground_truth/SIGL" "../models/SIGL" "../logs/SIGL")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done

# Clean working dir
dir1="../tmpTrainDir"
dir2="../tmpTestDir"
if [ -d "$dir1" ]; then
    rm -rf "$dir1"
fi

if [ -d "$dir2" ]; then
    rm -rf "$dir2"
fi

nohup mkdir ../tmpTrainDir &
nohup mkdir ../tmpTestDir &

wait

# Process interpreted train data
nohup python preprocess_custom.py \
 -files \
 /root/AirTag/sigl_data/7zip-1.log.interpret.log \
 /root/AirTag/sigl_data/7zip-2.log.interpret.log \
 /root/AirTag/sigl_data/7zip-3.log.interpret.log \
 /root/AirTag/sigl_data/7zip-4.log.interpret.log \
 /root/AirTag/sigl_data/7zip-5.log.interpret.log \
 /root/AirTag/sigl_data/7zip-6.log.interpret.log \
 /root/AirTag/sigl_data/dropbox-1.log.interpret.log \
 /root/AirTag/sigl_data/dropbox-2.log.interpret.log \
 /root/AirTag/sigl_data/dropbox-3.log.interpret.log \
 /root/AirTag/sigl_data/dropbox-4.log.interpret.log \
 /root/AirTag/sigl_data/dropbox-5.log.interpret.log \
 /root/AirTag/sigl_data/dropbox-6.log.interpret.log \
 /root/AirTag/sigl_data/filezilla-1.log.interpret.log \
 /root/AirTag/sigl_data/filezilla-2.log.interpret.log \
 /root/AirTag/sigl_data/filezilla-3.log.interpret.log \
 /root/AirTag/sigl_data/filezilla-4.log.interpret.log \
 /root/AirTag/sigl_data/filezilla-5.log.interpret.log \
 /root/AirTag/sigl_data/filezilla-6.log.interpret.log \
 /root/AirTag/sigl_data/firefox-1.log.interpret.log \
 /root/AirTag/sigl_data/firefox-2.log.interpret.log \
 /root/AirTag/sigl_data/firefox-3.log.interpret.log \
 /root/AirTag/sigl_data/firefox-4.log.interpret.log \
 /root/AirTag/sigl_data/firefox-5.log.interpret.log \
 /root/AirTag/sigl_data/firefox-6.log.interpret.log \
 /root/AirTag/sigl_data/geany-1.log.interpret.log \
 /root/AirTag/sigl_data/geany-2.log.interpret.log \
 /root/AirTag/sigl_data/geany-3.log.interpret.log \
 /root/AirTag/sigl_data/geany-4.log.interpret.log \
 /root/AirTag/sigl_data/geany-5.log.interpret.log \
 /root/AirTag/sigl_data/geany-6.log.interpret.log \
 /root/AirTag/sigl_data/gimp-1.log.interpret.log \
 /root/AirTag/sigl_data/gimp-2.log.interpret.log \
 /root/AirTag/sigl_data/gimp-3.log.interpret.log \
 /root/AirTag/sigl_data/gimp-4.log.interpret.log \
 /root/AirTag/sigl_data/gimp-5.log.interpret.log \
 /root/AirTag/sigl_data/gimp-6.log.interpret.log \
 /root/AirTag/sigl_data/pwsafe-1.log.interpret.log \
 /root/AirTag/sigl_data/pwsafe-2.log.interpret.log \
 /root/AirTag/sigl_data/pwsafe-3.log.interpret.log \
 /root/AirTag/sigl_data/pwsafe-4.log.interpret.log \
 /root/AirTag/sigl_data/pwsafe-5.log.interpret.log \
 /root/AirTag/sigl_data/pwsafe-6.log.interpret.log \
 /root/AirTag/sigl_data/qbittorrent-1.log.interpret.log \
 /root/AirTag/sigl_data/qbittorrent-2.log.interpret.log \
 /root/AirTag/sigl_data/qbittorrent-3.log.interpret.log \
 /root/AirTag/sigl_data/qbittorrent-4.log.interpret.log \
 /root/AirTag/sigl_data/qbittorrent-5.log.interpret.log \
 /root/AirTag/sigl_data/qbittorrent-6.log.interpret.log \
 /root/AirTag/sigl_data/skype-1.log.interpret.log \
 /root/AirTag/sigl_data/skype-2.log.interpret.log \
 /root/AirTag/sigl_data/skype-3.log.interpret.log \
 /root/AirTag/sigl_data/skype-4.log.interpret.log \
 /root/AirTag/sigl_data/skype-5.log.interpret.log \
 /root/AirTag/sigl_data/skype-6.log.interpret.log \
 -o_dir /root/AirTag/tmpTrainDir &

# Process interpreted test data
nohup python preprocess_custom.py \
 -files \
 /root/AirTag/sigl_mal/teamviewer-mal.interpret.log \
 -gt \
 /root/AirTag/sigl_mal/gt_teamviewer-mal.interpret.log.txt \
 -o_dir /root/AirTag/tmpTestDir &

wait

# Concatenate train and test data
nohup python concatenate_sigl_sets.py \
 -dir /root/AirTag/tmpTrainDir\
 -out /root/AirTag/tmpTrainDir/train_preprocessed_teamviewer &

wait

# Tag train, test logs given pattern (e.g 192.168.33.12)
nohup python tag_positive_logs.py \
 -file /root/AirTag/tmpTrainDir/train_preprocessed_teamviewer \
 -pattern 192.168.33.12 &

nohup python tag_positive_logs.py \
 -file /root/AirTag/tmpTestDir/teamviewer-mal.interpret.log \
 -pattern 192.168.33.12 &

wait

# Create vocab, train file
nohup python ../create_vocab_atlas.py  \
 -path /root/AirTag/tmpTrainDir/train_preprocessed_teamviewer \
 -t_name train_teamviewer \
 -v_name vocab_sigl_teamviewer.txt &

# Create test file, dont need its vocab
nohup python ../create_vocab_atlas.py  \
 -path /root/AirTag/tmpTestDir/teamviewer-mal.interpret.log \
 -t_name test_teamviewer \
 -v_name vocab_sigl_teamviewer_test.txt &

wait

rm /root/AirTag/tmpTestDir/vocab_sigl_teamviewer_test.txt 

# Extract benign file
input_file="../tmpTrainDir/train_teamviewer"
output_file="../tmpTrainDir/benign_teamviewer"
grep -v '+$' "$input_file" > "$output_file" & wait

# Create gt from test file
nohup python create_gt_from_processed_file.py \
 -file /root/AirTag/tmpTestDir/test_teamviewer \
 -out /root/AirTag/ground_truth/SIGL/teamviewer_number.npy &

wait

# Move to correct dir
mv ../tmpTrainDir/train_teamviewer \
 ../tmpTrainDir/benign_teamviewer \
 ../tmpTrainDir/vocab_sigl_teamviewer.txt \
 ../tmpTrainDir/train_preprocessed_teamviewer \
 ../tmpTestDir/test_teamviewer \
 ../training_data/SIGL/ &

wait

rm -rf ../tmpTrainDir &
rm -rf ../tmpTestDir &

wait

nohup python ../create_pretraining_data.py  \
 --input_file=../training_data/SIGL/train_teamviewer \
 --output_file=../training_data/SIGL/teamviewer.tfrecord \
 --vocab_file=../training_data/SIGL/vocab_sigl_teamviewer.txt \
 --do_lower_case=True \
 --max_seq_length=32  \
 --max_predictions_per_seq=20  \
 --masked_lm_prob=0.15  \
 --random_seed=12345  \
 --dupe_factor=5 &

wait

  nohup python -u ../run_pretraining.py  \
  --input_file=../training_data/SIGL/teamviewer.tfrecord \
  --output_dir=../models/SIGL/teamviewer \
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
 --learning_rate=2e-5 > ../logs/SIGL/trainTeamviewer.log &

wait

# extract embeddings

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/SIGL/benign_teamviewer \
  --output_file=../embedding_data/SIGL/benign_teamviewer.json  \
  --vocab_file=../training_data/SIGL/vocab_sigl_teamviewer.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/SIGL/teamviewer/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/SIGL/extract_teamviewer.log &

wait

  nohup python -u ../extract_multi_process2.py \
  --input_file=../training_data/SIGL/test_teamviewer \
  --output_file=../embedding_data/SIGL/test_teamviewer.json  \
  --vocab_file=../training_data/SIGL/vocab_sigl_teamviewer.txt \
  --bert_config_file=../uncased_L-6_H-128_A-2/bert_config.json \
  --init_checkpoint=../models/SIGL/teamviewer/model.ckpt-10000 \
  --layers=-1  \
  --gpu=0  \
  --max_seq_length=32 \
  --batch_size=2048 > ../logs/SIGL/extract_teamviewer_test.log &

wait

nohup python -u evaluate_sigl_data.py -flag 1 -nu 0.1 -gama 0.1 -gpu 0 -suffix SIGL > ../logs/SIGL/teamviewer.log &

wait
