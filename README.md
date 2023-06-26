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


We recommend trying MiniGPT4 first since it has quite a strong abstract understanding ability derived from its Large Language Model decoder (Vicuna), which facilitates humour generation and understanding.

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

**1. Modify the config file**

Put the downloaded weight in the directory: MODEL_PATH

minigpt4/eval_configs/minigpt4_eval.yaml

```
model:
  arch: mini_gpt4
  model_type: pretrain_vicuna
  freeze_vit: True
  freeze_qformer: True
  max_txt_len: 160
  end_sym: "###"
  low_resource: True
  prompt_path: "prompts/alignment.txt"
  prompt_template: '###Human: {} ###Assistant: '
  ckpt: 'downloaded_weight_path'
```


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

### Training the model


**1. Process the Oxford HIC dataset**

CLIPCap encodes images with CLIP and saves visual features to speed up training. 

Put the downloaded data in the directory YOUR_RAW_DATA_PATH, and specify the output directory YOUR_DATA_SAVE_DIR for processed data. Then run:

```
python clipcap/parse_humor.py --data_path YOUR_RAW_DATA_PATH --output_dir YOUR_DATA_SAVE_DIR --use_cuda
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

