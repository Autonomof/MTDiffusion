import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

sys.path.append('./')
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing
# from ema_pytorch import EMA
from Noam_Scheduler import Noam_Scheduler
torch.multiprocessing.set_sharing_strategy('file_system')

amp = None

from datasets import LMDBDataset
# from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler
from bit_diffusion.conditional_mtdiffusion.mtdiffusion import MTDiffusion, Unet

clas_dict: dict = {
    "DogBark": 0,
    "Footstep": 1,
    "GunShot": 2,
    "Keyboard": 3,
    "MovingMotorVehicle": 4,
    "Rain": 5,
    "Sneeze_Cough": 6,
}
# label = list(clas_dict.keys())[0]
label = 'all'
weight = 0.2



def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)
    # criterion = nn.CrossEntropyLoss()
    zero_pad = torch.nn.ZeroPad2d((0, 88 - 86, 0, 24 - 20))
    for i, (bottom, class_id, salience, file_name) in enumerate(loader):
        model.zero_grad()
        class_id = torch.FloatTensor(list(list(class_id))).long().unsqueeze(1)
        # salience = torch.FloatTensor(list(map(eval, list(salience)))).unsqueeze(1)

        bottom = bottom.to(device)   #(B,64,20,86)
        class_id = class_id.to(device)
        # salience = salience.to(device)

        # target = bottom

        # # print(target[0])
        # out, _ = model(bottom, label_condition=class_id)
        # # out, _ = model(bottom, label_condition=class_id, salience_condition=salience)
        #
        # loss = criterion(out, target)
        # # print(loss)
        # loss.backward()
        # bottom1 = bottom.unsqueeze(1)
        bottom = zero_pad(bottom)   #(B,64,24,88)
        # print(bottom.shape,bottom1.shape,bottom[1,0,21,:],bottom[1,0,15,:])
        print(torch.std_mean(bottom,dim=(1,2,3)))
        loss_img, loss_noise = model(bottom,class_id)
        loss = loss_img + loss_noise * weight
        loss.backward()
        # print(loss)
        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # _, pred = out.max(1)
        # correct = (pred == target).float()
        # accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'{label}_epoch: {epoch + 1};loss: {loss.item():.3f}; '
                f'lr: {lr:.5f}'
            )
        )


# class PixelTransform:
#     def __init__(self):
#         pass
#
#     def __call__(self, input):
#         ar = np.array(input)
#
#         return torch.from_numpy(ar).long()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=2000)
    # parser.add_argument('--hier', type=str, default='bottom')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--path', type=str, default=f'vqvae-code/{label}')
    parser.add_argument('--outdir', type=str, default=f'checkpoint/dual_unet_diffusion_{label}_v3_{weight}')
    parser.add_argument('--weight', type=float, default=0.2)
    args = parser.parse_args()

    print(args)


    # save_dir = os.path.join()
    os.makedirs(args.outdir, exist_ok=True)
    save_dir = args.outdir
    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False
    )
    ckpt = {}
    start_point = 0
    if args.ckpt is None and os.path.exists(args.outdir):
        ckpts = os.listdir(args.outdir)
        if len(ckpts) > 0:
            ckpt = sorted(ckpts)[-1]
            start_point = os.path.basename(ckpt)
            start_point = int(start_point.split('.')[0])
            ckpt = os.path.join(os.path.abspath(args.outdir), ckpt)
            print(ckpt)
            ckpt = torch.load(ckpt)

    elif args.ckpt is not None:
        # _, start_point = args.ckpt.split('_')
        start_point = os.path.basename(args.ckpt)
        start_point = int(start_point.split('.')[0])

        ckpt = torch.load(args.ckpt)
        # args = ckpt['args']

    os.makedirs(args.outdir, exist_ok=True)
    unet = Unet(
        dim=64,
        channels=64,
        dim_mults=(1, 2, 4),
    ).cuda()
    model = BitDiffusion(
        unet,
        image_size=(24, 88),
        timesteps=1000,
        time_difference=0,
        # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
        use_ddim=False  # use ddim
    ).cuda()
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    # print(model)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    #
    # if amp is not None:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.99),
        eps=1.0e-9
    )
    scheduler = Noam_Scheduler(
        optimizer=optimizer,
        warmup_steps=1000,
    )

    # scheduler = None
    # if args.sched == 'cycle':
    #     scheduler = CycleScheduler(
    #         optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
    #     )

    for i in range(start_point, start_point + args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if i % 20 < 2:
            torch.save(
                {'model': model.module.state_dict(), 'args': args},
                f'{save_dir}/{str(i + 1).zfill(5)}.pt',
            )
