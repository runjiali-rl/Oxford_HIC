# Oxford_HIC
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/1234.56789)

Coming Soon


## Online Demo

Click the image to chat with models trained on Oxford HIC around your images
[![demo](figs/online_demo.png)](https://minigpt-4.github.io)


## Examples
  |   |   |
:-------------------------:|:-------------------------:
![find wild](figs/examples/wop_2.png) |  ![write story](figs/examples/ad_2.png)
![solve problem](figs/examples/fix_1.png)  |  ![write Poem](figs/examples/rhyme_1.png)



## Introduction
This paper presents Oxford HIC, a large-scale dataset for humour generation and understanding. Humour is an abstract, subjective, and context-dependent cognitive construct involving several cognitive factors, making it a challenging task to generate and interpret. Hence, humour generation and understanding can serve as a new task for evaluating the ability of deep-learning methods to process abstract and subjective information.
Due to the scarcity of data, humour-related generation tasks such as captioning remain under-explored.
To address this gap, Oxford HIC offers approximately 2.9M image-text pairs with humour scores to train a generalizable humour captioning model.
Contrary to existing captioning datasets, Oxford HIC features a wide range of emotional and semantic diversity resulting in out-of-context examples that are particularly conducive to generating humour. Moreover, Oxford HIC is curated devoid of offensive content.
We also show how Oxford HIC can be leveraged for evaluating the humour of a generated text. 
Through explainability analysis of the trained models, we identify the visual and linguistic cues influential for evoking humour prediction (and generation). We observe qualitatively that these cues are aligned with the benign violation theory of humour in cognitive psychology.


![overview](figs/overview.png)

## Dataset download
Download the dataset from [Oxford_HIC](https://drive.google.com/file/d/1lwI3T81QtbVtIcJ7jsRxdxBI1bh_4wy7/view?usp=sharing) 


## Demo weight download
Download the pre-trained checkpoints on MiniGPT4 and CLIPCap

|                                MiniGPT4                                |                               CLIPCap                              |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Downlad](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1lwI3T81QtbVtIcJ7jsRxdxBI1bh_4wy7/view?usp=sharing) 


We recommend trying MiniGPT4 first since it has quite a strong abstract understanding ability derived from its Large Language Model decoder (Vicuna), which facilitates humour generation and understanding. Suppose you download the pth file in the path MODEL_DIR

## Model: MiniGPT4
### Preparation

**1. Prepare the code and the environment**

Git clone our repository, create a Python environment and activate it via the following command

```bash
git clone git@github.com:liguang0115/Oxford_HIC.git
cd Oxford_HIC/minigpt4
conda env create -f environment.yml
conda activate minigpt4
```

### Launching demo locally
**1. Prepare the pre-trained Vicuna weights**

Download the pre-trained checkpoints according to the Vicuna model you prepare.

|                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Downlad](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) 
 
The humour generator of MiniGPT-4 is built on the v0 version of Vicuna-7B.
Please refer to the instruction [here](minigpt4/PrepareVicuna.md) 
to prepare the Vicuna weights.
The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[minigpt4/minigpt4/configs/models/minigpt4.yaml](minigpt4/minigpt4/configs/models/minigpt4.yaml) at Line 16.


**2. Load the pre-trained MiniGPT-4 checkpoint**


Then, set the path to the pre-trained checkpoint: MODEL_DIR in the evaluation config file 
in [minigpt4/eval_configs/minigpt4_eval.yaml](minigpt4/eval_configs/minigpt4_eval.yaml#L10) at Line 11. 


**3. Launching demo**

Try out the demo [minigpt4/demo.py](minigpt4/demo.py) on your local machine by running

```
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

### Training
The training of MiniGPT-4 contains two alignment stages.

**1. Download pretraining weight**

We directly use the pre-training weight from the original MiniGPT4 repository
which could be downloaded from
[here (13B)](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link) or [here (7B)](https://drive.google.com/file/d/1HihQtCEXUyBM1i9DQbaK934wW3TZi-h5/view?usp=share_link).

**2. Process the Oxford HIC dataset**


**3. Humour finetuning stage**

In the second stage, we use a small high-quality image-text pair dataset created by ourselves
and convert it to a conversation format to further align MiniGPT-4.
To download and prepare our second-stage dataset, please check our 
[second stage dataset preparation instruction](dataset/README_2_STAGE.md).
To launch the second stage alignment, 
first specify the path to the checkpoint file trained in stage 1 in 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage2_finetune.yaml).
You can also specify the output path there. 
Then, run the following command. In our experiments, we use 1 A100.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

After the second stage alignment, MiniGPT-4 is able to talk about the image coherently and user-friendly. 

## Model: CLIPCap
### Preparation

**1. Prepare the code and the environment**

Git clone our repository, create a Python environment and activate it via the following command

```bash
git clone git@github.com:liguang0115/Oxford_HIC.git
cd Oxford_HIC/clipcap
conda env create -f environment.yml
conda activate humor
```






### Launching demo locally
Put the downloaded weight in the directory: MODEL_PATH

Put the image that you want to generate the joke on in the path IMAGE_PATH

then run:

```
python inference.py --model_path MODEL_PATH --image_path IMAGE_PATH
```

or if you want to use a prompt P, run:

```
python inference.py --model_path MODEL_PATH --image_path IMAGE_PATH --prompt P
```

### Training


**1. Process the Oxford HIC dataset**

CLIPCap encodes images with CLIP and saves visual features to speed up training. 

Put the downloaded data in the directory YOUR_RAW_DATA_PATH, and specify the output directory YOUR_DATA_SAVE_DIR for processed data. Then run:

```
python parse_humor.py --data_path YOUR_RAW_DATA_PATH --output_dir YOUR_DATA_SAVE_DIR --use_cuda
```

The output processed dataset will be in a pkl file.

**2. Training**

Train the model with a distributed system on your machine by running

```
torchrun --nproc_per_node=NUM_GPUS train.py --data YOUR_DATA_SAVE_DIR --out_dir CKPT_OUTPUT_DIR --epochs EPOCH --bs BATCH_SIZE --lr LR
```
CKPT_OUTPUT_DIR: the directory to save your model weights

BATCH_SIZE: the batch size of data for training 

LR: the initial learning rate after warm-up

