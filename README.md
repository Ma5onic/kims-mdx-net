# KUIELab-MDX-Net

- This is a modified KUIELab-MDX-Net to train a vocal model with just the vocals and instrumental (no bass, drums or other stem needed)

## 0. Environment

- I used Ubuntu 18.04
- cuda-able GPU (>= 2080ti) (I used a Nvidia T4 from aws which has 16gb vram)
- wandb for logging

## 1. Installation

```bash
conda env create -f conda_env_gpu.yaml -n mdx-net
conda activate mdx-net
pip install -r requirements.txt

sudo apt-get install soundstretch
```

## 2. Preparing data and files

For training data you need 3 audio files per track and folder. One called vocals.wav that contains the acapella, One called other.wav that contains the instrumental, One called mixture.wav that contains the acapella from vocals.wav and instrumental from other.wav mixed together. Each track in your dataset needs its own folder.
For validation data you only need 2 audio files per track and folder, one called vocals.wav and one called mixture.wav, Do not use the same data for training and validation !!

Inside the data folder there is a folder called train, put your training track folders and validation track folders inside the train folder.

Sign up to https://wandb.ai/site . once you are logged in go to https://wandb.ai/settings and copy your API key.

Open the .env.example file and after wandb_api_key= paste your API key then after data_dir= enter your data folder path (not train folder path)

Windows it will look something like this data_dir=F:\mdx-net\data

Ubuntu it will look something like this data_dir=/home/ubuntu/mdx-net/data

Then save the file but rename it to .env

Open configs/datamodule/musdb18_hq.yaml
 
Under validation_set: is where you put the names of the folders of your validation set (delete the ones that are already there)

## 3. Training

Make sure you are in the mdx-net directory in the command line and you have run conda activate mdx-net if you are using ubuntu

Copy the command below to start training

python run.py experiment=multigpu_vocals model=ConvTDFNet_vocals

## 4. Separation
https://github.com/kuielab/mdx-net-submission/tree/leaderboard_B

## 5. Extra information

You might want to or need to change the batch size in configs/experiment/multigpu_vocals.yaml depending on your PC, the default batch size is 6 that works on a 16gb gpu

If you want to change how often validation is done go to mdx-net/configs/trainer/minimal.yaml and change check_val_every_n_epoch: 1 to the number you would like.

# ACKNOWLEDGEMENT
- https://github.com/kuielab/mdx-net for the code and https://github.com/ws-choi for everything he has helped me with 
- This repository is based on [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template)
- Also, facebook/[demucs](https://github.com/facebookresearch/demucs)
