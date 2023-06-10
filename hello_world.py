import torch, torchvision
print('Pytorch 版本', torch.__version__)
print('CUDA 是否可用',torch.cuda.is_available())

import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('MMCV版本', mmcv.__version__)
print('CUDA版本', get_compiling_cuda_version())
print('编译器版本', get_compiler_version())

import mmdet
print('mmdetection版本', mmdet.__version__)

import mmpose
print('mmpose版本', mmpose.__version__)

from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

def collect_env():
    env_info = collect_base_env()
    env_info['MMDetection'] = f'{mmdet.__version__}+{get_git_hash()[:7]}'
    retrun env_info

if __name__ == "__main__":
    for name, val in collect_env.items():
        print(f'{name}:{val}')