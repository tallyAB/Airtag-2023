# Airtag
Fork of [AirTag](https://github.com/dhl123/Airtag-2023) for reproduction and additional experiments

## Setup
- Please use the [Dockerfile](./Dockerfile) provided. Run an interactive session with runtime=nvidia for gpu pass-through to container.
**Note**: Requires [nvidia-docker](https://hub.docker.com/r/tensorflow/tensorflow) to be setup 
- Download data, models, and embedding vectors from [Drive](https://drive.google.com/drive/u/1/folders/1XAvQnCfo1J-OPXdkHYTrKWC3X_XJivS1).
- Unpack all directories and move them to the /root/AirTag

## Run
Change working directory
```
cd /root/AirTag/ablation_scripts
```
Reproduce Paper Results
```
bash effect_org.sh
```
Ablation 1: ATLAS Event logs only
```
bash effect_ablation_one.sh
```
Ablation 2: ATLAS Event + DNS logs
```
bash effect_ablation_two.sh
```
AirTag on SIGL data
```
bash effect_sigl.sh
```
Results in [logs](./logs/)
