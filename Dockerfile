FROM tensorflow/tensorflow:1.11.0-gpu-py3

WORKDIR /root/AirTag

# Requisites
RUN apt-get update && apt-get install python3-tk -y && apt-get install git -y && pip install thundersvm

# Download repository
RUN cd /root \
&& git clone https://github.com/tallyAB/Airtag-2023.git \
&& rmdir AirTag \
&& mv Airtag-2023 AirTag