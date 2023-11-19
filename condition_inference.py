import sys
from typing import List
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod
import numpy

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import math
import time
import datetime
import torch
from tqdm import tqdm
import soundfile as sf
from einops import rearrange, reduce, repeat

# from vqvae import VQVAE
# from pixelsnail import PixelSNAIL
from bit_diffusion.conditional_bit_diffusion.bit_diffusion_v3 import BitDiffusion, Unet
from HiFiGanWrapper import HiFiGanWrapper
from rvqvae import SoundStream
import sys

class SoundSynthesisModel(ABC):
    @abstractmethod
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        raise NotImplementedError


BITS = 9
MaxNum = 2 ** BITS - 1
# weight = '_0'
weight = sys.argv[2]


clas_dict: dict = {
    "DogBark": 0,
    "Footstep": 1,
    "GunShot": 2,
    "Keyboard": 3,
    "MovingMotorVehicle": 4,
    "Rain": 5,
    "Sneeze_Cough": 6,
}
# label_id = 6
# label = list(clas_dict.keys())[label_id]
label = sys.argv[1]
label_id = clas_dict[label]

class DCASE2023FoleySoundSynthesis:
    def __init__(
            self, number_of_synthesized_sound_per_class: int = 100, batch_size: int = 16
    ) -> None:
        self.number_of_synthesized_sound_per_class: int = (
            number_of_synthesized_sound_per_class
        )
        self.batch_size: int = batch_size
        # self.class_id_dict: dict = {
        #     0: 'DogBark',
        #     # 1: 'Footstep',
        #     # 2: 'GunShot',
        #     # 3: 'Keyboard',
        #     # 4: 'MovingMotorVehicle',
        #     # 5: 'Rain',
        #     # 6: 'Sneeze',
        # }
        self.sr: int = 22050
        self.save_dir: str = f"./dual_synthesized_rvq_mtdiffusion_weight{weight}"
        print('save',self.save_dir)

    def synthesize(self, synthesis_model: SoundSynthesisModel) -> None:
        # for sound_class_id in self.class_id_dict:
        if not os.path.exists(self.save_dir):
            os.system(f'mkdir -p {self.save_dir}')
        sample_number: int = 1
        save_category_dir: str = (
            f'{self.save_dir}/{label}'
        )
        os.makedirs(save_category_dir, exist_ok=True)
        for _ in tqdm(
                range(
                    math.ceil(
                        self.number_of_synthesized_sound_per_class / self.batch_size
                    )
                ),
                desc=f"Synthesizing {label}",
        ):
            global label_id
            label_id_input = torch.tensor([label_id] * self.batch_size).to('cuda')
            print(f'generate {len(label_id_input)}')
            synthesized_sound_list: list = synthesis_model.synthesize_sound(
                label_id_input, self.batch_size
            )
            for synthesized_sound in synthesized_sound_list:
                if sample_number <= self.number_of_synthesized_sound_per_class:
                    sf.write(
                        f"{save_category_dir}/{str(sample_number).zfill(4)}.wav",
                        synthesized_sound,
                        samplerate=self.sr,
                    )
                    sample_number += 1


# ================================================================================================================================================
class BaseLineModel(SoundSynthesisModel):
    def __init__(
            self, diffusion_checkpoint: str, vqvae_snail_checkpoint: str, timesteps: int = 500,use_ddim = True,
    ) -> None:
        super().__init__()
        # self.pixel_snail = PixelSNAIL(
        #     [20, 86],
        #     512,
        #     256,
        #     5,
        #     4,
        #     4,
        #     256,
        #     dropout=0.1,
        #     n_cond_res_block=3,
        #     cond_res_channel=256,
        # )
        # self.pixel_snail.load_state_dict(
        #     torch.load(pixel_snail_checkpoint, map_location='cpu')['model']
        # )
        # self.pixel_snail.cuda()
        # self.pixel_snail.eval()
        self.unet = Unet(
            dim=64,
            channels=64,
            dim_mults=(1, 2, 4),
        )
        self.bitdiffusion = BitDiffusion(
            self.unet,
            image_size=(24, 88),
            timesteps=timesteps,
            time_difference=0,
            # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
            use_ddim=use_ddim  # use ddim
        )
        self.bitdiffusion.load_state_dict(
            torch.load(diffusion_checkpoint, map_location='cpu')['model']
        )
        self.bitdiffusion.cuda()
        self.bitdiffusion.eval()

        # self.vqvae = VQVAE()
        # self.vqvae.load_state_dict(
        #     torch.load(vqvae_snail_checkpoint, map_location='cpu')
        # )
        # self.vqvae.cuda()
        # self.vqvae.eval()
        num_quantizers = 4
        self.model = SoundStream(n_q=num_quantizers, codebook_size=512, stride=4).cuda()
        self.model.load_state_dict(torch.load(vqvae_snail_checkpoint, map_location='cpu'))
        self.model.eval()

        self.hifi_gan = HiFiGanWrapper(
            './checkpoint/hifigan/g_00935000',
            './checkpoint/hifigan/hifigan_config.json',
        )

    @torch.no_grad()
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        audio_list: List[ndarray] = list()

        feature_shape: list = [20, 86]
        bottoms = self.bitdiffusion.sample(class_id,batch_size=number_of_sounds)
        bottoms = bottoms[:,:, 0:20, 0:86]
        codes_bottom = self.model.quantizer(bottoms)[1]
        print(codes_bottom.shape)
        pred_mel = self.model.decode_ids(codes_bottom).detach()
        for j, mel in enumerate(pred_mel):
            audio_list.append(
                numpy.concatenate((self.hifi_gan.generate_audio_by_hifi_gan(mel), numpy.zeros(88200 - 88064))))
        return audio_list


# ===============================================================================================================================================
if __name__ == '__main__':
    start = time.time()
    number_of_synthesized_sound_per_class =500
    batch_size = 500
    # diffusion_checkpoint = None
    timesteps = 500
    use_ddim = True
    # weight = '_0.1'
    vqvae_checkpoint = './checkpoint/4quant_rvqvae/vqvae_750.pt'
    if weight in ['2','4','6','8']:
        weight=int(weight)/10

    weight = '_' + str(weight)
    ckpt_dir = f'./checkpoint/dual_unet_diffusion_all_v3_0.2'
    print(ckpt_dir)
    dcase_2023_foley_sound_synthesis = DCASE2023FoleySoundSynthesis(
        number_of_synthesized_sound_per_class, batch_size
    )
    # ckpt = 'checkpoint/diffusion_all_v3_0.2/01981.pt'
    ckpt = None
    if ckpt is None and os.path.exists(ckpt_dir):
        ckpts = os.listdir(ckpt_dir)
        ckpt = sorted(ckpts)[-1]
        ckpt =os.path.join(os.path.abspath(ckpt_dir),ckpt)
    if ckpt is None:
        raise OSError
    print(ckpt)
    dcase_2023_foley_sound_synthesis.synthesize(
        synthesis_model=BaseLineModel(ckpt, vqvae_checkpoint, timesteps,use_ddim)
    )
    print(str(datetime.timedelta(seconds=time.time() - start)))
