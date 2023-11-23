import os

def rename_files_in_folder(folder_path):
    # 获取文件夹中的所有文件
    dirnames = os.listdir(folder_path)
    print(dirnames)
    for dirname in dirnames:
        sub_folder_path = os.path.join(folder_path, dirname)
        files = os.listdir(sub_folder_path)
        # 对文件按照它们在文件夹中的顺序进行排序
        files.sort()

        # 遍历文件并进行重新命名
        for index, file_name in enumerate(files):
            # 构建新的文件名，例如：new_name_1, new_name_2, ...
            new_name = f"{str(index).zfill(2)}.wav"

            # 构建文件的完整路径
            old_file_path = os.path.join(sub_folder_path, file_name)
            new_file_path = os.path.join(sub_folder_path, new_name)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_name}")

# 调用函数并传递文件夹路径
folder_path = "/mnt/autonomf_4T/dcase7/code/MTDiffusion/audio_samples/Baseline"
rename_files_in_folder(folder_path)
