import sys
from typing import List
from numpy import ndarray
import numpy
from torch import Tensor
from abc import ABC, abstractmethod

import os
# import argparse
import math
import time
import datetime
import torch
from tqdm import tqdm
import soundfile as sf
from einops import rearrange, reduce, repeat

from vqvae import VQVAE
# from pixelsnail import PixelSNAIL
from bit_diffusion.bit_diffusion.bit_diffusion import BitDiffusion, Unet
from HiFiGanWrapper import HiFiGanWrapper


class SoundSynthesisModel(ABC):
    @abstractmethod
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        raise NotImplementedError


# BITS = 9
# MaxNum = 2 ** BITS - 1

clas_dict: dict = {
    "DogBark": 0,
    "Footstep": 1,
    "GunShot": 2,
    "Keyboard": 3,
    "MovingMotorVehicle": 4,
    "Rain": 5,
    "Sneeze_Cough": 6,
}
# label = list(clas_dict.keys())[]
label = sys.argv[1]

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
        self.save_dir: str = "./synthesized"

    def synthesize(self, synthesis_model: SoundSynthesisModel) -> None:
        # for sound_class_id in self.class_id_dict:
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
            synthesized_sound_list: list = synthesis_model.synthesize_sound(
                1 , self.batch_size
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
            self, diffusion_checkpoint: str, vqvae_snail_checkpoint: str, timesteps: int = 500,
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
            dim=48,
            channels=64,
            dim_mults=(1, 2,4),
        )
        self.bitdiffusion = BitDiffusion(
            self.unet,
            image_size=(24, 88),
            timesteps=timesteps,
            time_difference=0,
            # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
            use_ddim=True  # use ddim
        )
        self.bitdiffusion.load_state_dict(
            torch.load(diffusion_checkpoint, map_location='cpu')['model']
        )
        self.bitdiffusion.cuda()
        self.bitdiffusion.eval()

        self.vqvae = VQVAE()
        self.vqvae.load_state_dict(
            torch.load(vqvae_snail_checkpoint, map_location='cpu')
        )
        self.vqvae.cuda()
        self.vqvae.eval()

        self.hifi_gan = HiFiGanWrapper(
            './checkpoint/hifigan/g_00935000',
            './checkpoint/hifigan/hifigan_config.json',
        )

    @torch.no_grad()
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        audio_list: List[ndarray] = list()

        feature_shape: list = [20, 86]
        # vq_token: Tensor = torch.zeros(
        #     number_of_sounds, *feature_shape, dtype=torch.int64
        # ).cuda()
        # cache = dict()

        # for i in tqdm(range(feature_shape[0]), desc="pixel_snail"):
        #     for j in range(feature_shape[1]):
        #         out, cache = self.bitdiffusion(
        #             vq_token[:, : i + 1, :],
        #             label_condition=torch.full([number_of_sounds, 1], int(class_id))
        #             .long()
        #             .cuda(),
        #             cache=cache,
        #         )
        #         prob: Tensor = torch.softmax(out[:, :, i, j], 1)
        #         vq_token[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)
        #
        # pred_mel = self.vqvae.decode_code(vq_token).detach()
        # for j, mel in enumerate(pred_mel):
        #     audio_list.append(self.hifi_gan.generate_audio_by_hifi_gan(mel))
        # return audio_list
        bottoms = self.bitdiffusion.sample(batch_size=number_of_sounds).permute(0,2,3,1)
        print(bottoms.shape)
        bottoms = bottoms[:,0:20,0:86,:]
        codes_bottom=self.vqvae.quantize_b(bottoms)[2]
        print(codes_bottom.shape)
        # print(vq_tokens)
        # vq_tokens = vq_tokens.long().squeeze(1)
        # print(vq_tokens[:,feature_shape])
        # print(vq_tokens.shape, vq_tokens)
        # print(vq_tokens.shape)
        # print(vq_tokens[:, 20, :], vq_tokens[:, :, 86])
        # vq_tokens = vq_tokens[:, :feature_shape[0], :feature_shape[1]]
        # print(vq_tokens[:, 19, :], vq_tokens[:, :, 85])
        # print(vq_tokens.shape, vq_tokens)
        pred_mel = self.vqvae.decode_code(codes_bottom).detach()
        for j, mel in enumerate(pred_mel):
            audio_list.append(
                numpy.concatenate((self.hifi_gan.generate_audio_by_hifi_gan(mel), numpy.zeros(88200 - 88064))))
        return audio_list


# ===============================================================================================================================================
if __name__ == '__main__':
    start = time.time()
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--vqvae_checkpoint', type=str, default='./checkpoint/vqvae/vqvae_800.pt'
    # )
    # parser.add_argument(
    #     '-t','--timesteps', type=int, default=500,
    # )
    # parser.add_argument(
    #     '--diffusion_checkpoint',
    #     type=str,
    #     # default='./checkpoint/diffusion_timedifference0.2/05705.pt',
    #     # default='./checkpoint/diffusion/25555.pt',
    # )
    # parser.add_argument(
    #     '--number_of_synthesized_sound_per_class', type=int, default=100
    # )
    # parser.add_argument('--batch_size', type=int, default=100)
    # args = parser.parse_args()
    number_of_synthesized_sound_per_class = 100
    batch_size = 100
    timesteps = 150
    print(timesteps)
    vqvae_checkpoint = './checkpoint/vqvae/vqvae_800.pt'
    dcase_2023_foley_sound_synthesis = DCASE2023FoleySoundSynthesis(
        number_of_synthesized_sound_per_class, batch_size
    )
    # ckpt = 'checkpoint/diffusion_MovingMotorVehicle/01961.pt'
    ckpt=None
    if ckpt is None and os.path.exists(f'./checkpoint/diffusion_{label}'):
        ckpts = os.listdir(f'checkpoint/diffusion_{label}')
        ckpt = sorted(ckpts)[-1]
        ckpt =os.path.join(os.path.abspath(f'checkpoint/diffusion_{label}'),ckpt)
        print(ckpt)
    if ckpt is None:
        raise OSError
    dcase_2023_foley_sound_synthesis.synthesize(
        synthesis_model=BaseLineModel(ckpt, vqvae_checkpoint, timesteps)
    )
    print(str(datetime.timedelta(seconds=time.time() - start)))
