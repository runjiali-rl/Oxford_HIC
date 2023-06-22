import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import pandas as pd


def main(clip_model_type: str):
    device = torch.device('cuda:2')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./humor_{clip_model_name}_single_demo.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    data = pd.read_csv('../humor_single_demo.csv')
    print("%0d captions loaded from csv " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        img_id = data.loc[i, "image_id"]
        filename = f"../images/{img_id}.jpg"
        try:
            image = io.imread(filename)
        except:
            print(filename, "does not exist")
            continue
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)
        data_dic = {
                'image_id': img_id,
                'caption': data.loc[i, 'caption'],
                'clip_embedding': i
                }
        all_captions.append(data_dic)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
