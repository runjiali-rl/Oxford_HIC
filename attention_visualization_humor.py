import clip
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from train_gpt2_clip import ClipCaptionModel

DEVICE = 3


def main(validation_id_path, save_dir):

    validation_ids = np.load(validation_id_path)

    model = ClipCaptionModel(prefix_length=10)
    model.load_state_dict(torch.load('clipcap_humor_clip.pth'))
    model = model.to(torch.device(f'cuda:{DEVICE}'))
    clip_model = model.clip_model
    _, preprocess = clip.load('ViT-B/32', device=torch.device(f'cuda:{DEVICE}'), jit=False)

    for validation_id in tqdm(validation_ids):
        im = Image.open(f"../datasets/images/{validation_id}.jpg")
        x = preprocess(im).to(torch.device(f'cuda:{DEVICE}'))

        att_mat = clip_model.get_image_attention_map(x.unsqueeze(0)).to(torch.device('cpu'))

        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat #+ residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        # result = (mask * im).astype("uint8")

        im_np = np.asarray(im)
        mask = (mask*255).astype('uint8')
        mask = (mask-mask.min())/(mask.max() - mask.min())*255

        mask = cv2.applyColorMap(mask.astype('uint8'), cv2.COLORMAP_JET)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        try:
            result = (0.5*mask+0.5*im_np).astype('uint8')
        except:
            print('image error')
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{save_dir}/{validation_id}_attention_map.jpg', result)
    # plt.imshow(result)
    # plt.show()

    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    #
    # ax1.set_title('Original')
    # ax2.set_title('Attention Map')
    # _ = ax1.imshow(im)
    # _ = ax2.imshow(result)

if __name__ == '__main__':
    validation_id_path = '../datasets/out_val_ids.npy'
    save_dir = '../datasets/clip_attention_map_humor'
    main(validation_id_path, save_dir)
