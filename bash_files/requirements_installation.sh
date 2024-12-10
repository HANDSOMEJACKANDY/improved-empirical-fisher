#!/bin/bash

# create virtual enviroment with conda
conda create -n iEF python=3.10.4
conda activate iEF

# start downloading packages
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scipy
pip install datasets
pip install transformers
pip install peft
pip install Jinja2
pip install wandb
