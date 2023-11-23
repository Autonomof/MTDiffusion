from pydub import AudioSegment
import os

def pad_wav_to_4s(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    for label in os.listdir(input_folder):
        # 获取输入文件夹中所有WAV文件的路径
        wav_files = [f for f in os.listdir(os.path.join(input_folder,label)) if f.endswith('.wav')]
        os.makedirs(os.path.join(output_folder,label), exist_ok=True)
        for wav_file in wav_files:
            input_path = os.path.join(os.path.join(input_folder,label), wav_file)
            output_path = os.path.join(os.path.join(output_folder,label), wav_file)

            # 读取WAV文件
            audio = AudioSegment.from_wav(input_path)

            # 获取当前音频的时长
            duration = len(audio)
            print(duration)
            # 计算需要补零的时长
            padding_duration = 4000 - duration

            # 如果需要补零，则进行补零操作
            if padding_duration > 0:
                silence = AudioSegment.silent(duration=padding_duration)
                padded_audio = audio + silence
            else:
                padded_audio = audio

            # 导出补零后的音频
            padded_audio.export(output_path, format="wav")

# 替换 'input_folder' 和 'output_folder' 为实际的输入和输出文件夹路径
pad_wav_to_4s('/mnt/autonomf_4T/dcase7/code/MTDiffusion/audio_samples/Baseline', '/mnt/autonomf_4T/dcase7/code/MTDiffusion/audio_samples/Baseline_new')
