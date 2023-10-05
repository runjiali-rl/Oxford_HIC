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