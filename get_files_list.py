import os
import re
from glob import glob

def get_my_files_list(data_path):
    IMG = glob(data_path + '/*.png')
    PRO = []
    for img in IMG:
        name = os.path.basename(img)
        PRO.append(name)
    return PRO

def get_my_files_prompts(data_path):
    IMG = glob(data_path + '/*.png')
    PRO = []
    for img in IMG:
        name = os.path.basename(img).split('.')[0].split('-')[1]
        name = re.sub('_', ' ', name)
        PRO.append(name)
    return PRO

if __name__ == "__main__":
    data_path = ''
    pro_list = get_my_files_list(data_path)
    pro_prompts = get_my_files_prompts(data_path)