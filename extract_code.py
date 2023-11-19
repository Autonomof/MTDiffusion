import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path
import pickle
import posixpath

import torch
from torch.utils.data import DataLoader
import lmdb
from tqdm import tqdm
import audio2mel
from datasets import get_dataset_filelist
# from vqvae import VQVAE
from datasets import CodeRow
from rvqvae import SoundStream
clas_dict: dict = {
    "DogBark": 0,
    "Footstep": 1,
    "GunShot": 2,
    "Keyboard": 3,
    "MovingMotorVehicle": 4,
    "Rain": 5,
    "Sneeze_Cough": 6,
}
label = list(clas_dict.keys())[1]


def embedding(idb_sub, embed):
    return torch.stack([embed[i] for i in idb_sub], dim=0)


def extract(lmdb_env, loader, model, device):
    index = 0
    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)
        # embed = model.quantize_b.embed.permute(1, 0)
        for img, class_id, salience, filename in pbar:
            img = img.to(device)
            quant, _, _ = model.encode(img)
            # idb = idb.permute(0,2,1).reshape(-1,20*86)
            # id_b=torch.stack([embedding(idb_sub,embed) for idb_sub in idb],dim=0)
            # print(id_b,id_b.shape)
            # # id_t = id_t.detach().cpu().numpy()
            # id_b = id_b.detach().cpu().numpy()
            # quant =quant.permute(0,3,1,2)
            quant = quant.detach().cpu().numpy()
            for c_id, sali, file, bottom in zip(class_id, salience, filename, quant):
                row = CodeRow(
                    bottom=bottom, class_id=c_id, salience=sali, filename=file
                )
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vqvae_checkpoint', type=str, default='./checkpoint/vqvae/vqvae_800.pt'
    )
    parser.add_argument('--name', type=str, default=f'vqvae-code/{label}_stride2')

    args = parser.parse_args()

    device = 'cuda'

    # train_file_list = get_dataset_filelist(label)
    #
    # train_set = audio2mel.Audio2Mel(
    #     train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
    # )
    #
    # loader = DataLoader(train_set, batch_size=16, sampler=None, num_workers=2)
    #
    # # for i, batch in enumerate(loader):l
    # #     mel, id, name = batch
    #
    # model = VQVAE()
    # model.load_state_dict(torch.load(args.vqvae_checkpoint, map_location='cpu'))
    # model = model.to(device)
    # model.eval()
    #
    # map_size = 100 * 1024 * 1024 * 1024
    #
    # env = lmdb.open(args.name, map_size=map_size)
    #
    # extract(env, loader, model, device)

# ### all class extract
# for label in list(clas_dict.keys()):
#     device = 'cuda'
#
#     train_file_list = get_dataset_filelist(label)
#
#     train_set = audio2mel.Audio2Mel(
#         train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
#     )
#
#     loader = DataLoader(train_set, batch_size=16, sampler=None, num_workers=2)
#
#     # for i, batch in enumerate(loader):l
#     #     mel, id, name = batch
#
#     model = VQVAE()
#     model.load_state_dict(torch.load('./checkpoint/vqvae/vqvae_800.pt', map_location='cpu'))
#     model = model.to(device)
#     model.eval()
#
#     map_size = 100 * 1024 * 1024 * 1024
#     if not os.path.exists(f'vqvae-code/{label}'):
#         os.makedirs(f'vqvae-code/{label}')
#     env = lmdb.open(f'vqvae-code/{label}', map_size=map_size)
#
#     extract(env, loader, model, device)

### all class extract together
device = 'cuda'
label = 'all'
train_file_list = get_dataset_filelist()

train_set = audio2mel.Audio2Mel(
    train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
)

loader = DataLoader(train_set, batch_size=4, sampler=None, num_workers=4)

# for i, batch in enumerate(loader):l
#     mel, id, name = batch

# model = VQVAE()
num_quantizers = 4
model = SoundStream(n_q=num_quantizers,codebook_size=512,stride=2).to(device)
model.load_state_dict(torch.load('./checkpoint/4quant_rvqvae_stride2//vqvae_551.pt', map_location='cpu'))
model = model.to(device)
model.eval()
map_size = 100 * 1024 * 1024 * 1024
if not os.path.exists(f'vqvae-code/{label}_stride2'):
    os.makedirs(f'vqvae-code/{label}_stride2')
env = lmdb.open(f'vqvae-code/{label}_stride2', map_size=map_size)

extract(env, loader, model, device)