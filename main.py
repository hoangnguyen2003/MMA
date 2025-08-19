import os
import sys

# Clear any problematic paths from PATH
path_dirs = os.environ.get('PATH', '').split(':')
clean_path = [d for d in path_dirs if not any(x in d for x in ['https', '//www.kaggle.com', 'gcr.io', 'kagglegym'])]
os.environ['PATH'] = ':'.join(clean_path)

# Set CUDA environment variables
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/opt/conda/lib:/usr/lib/x86_64-linux-gnu'

# Add to existing LD_LIBRARY_PATH if it exists
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/opt/conda/lib:/usr/lib/x86_64-linux-gnu'

# Create necessary symlinks - run this in a separate cell first
!ln -sf /usr/local/cuda/lib64/libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so 2>/dev/null || true
!ln -sf /usr/local/cuda/lib64/libcudart.so /opt/conda/lib/libcudart.so.11.0 2>/dev/null || true

print("Environment setup completed")
import torch
import argparse
import numpy as np
import logging
from dataset import *
from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from utils.logs import set_arg_log

logging.getLogger ().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, filename='MMA.log')

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor') 
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(seed) 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False  

        use_cuda = True

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())  
    
    set_seed(args.seed)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    dataLoader = MMDataLoader(args)

    train_loader = dataLoader['train']
    valid_loader = dataLoader['valid']
    test_loader = dataLoader['test']

    torch.autograd.set_detect_anomaly(True)
    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader)

    logging.info(f'Runing code on the {args.dataset} dataset.')
    set_arg_log(args)
    best_dict = solver.train_and_eval()

    logging.info(f'Training complete')
    logging.info('--'*50)
    logging.info('\n')