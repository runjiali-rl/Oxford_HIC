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

