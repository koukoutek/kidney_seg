import torch
import argparse
import os
import logging
import logging.config
import numpy as np
import random
import yaml
import warnings

from monai.utils import set_determinism
from monai.config.deviceconfig import print_config
from utils import evaluate_true_false
from train_contrast import train_contrast
from train_non_contrast import train_non_contrast
from pathlib import Path

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
CUDA_LAUNCH_BLOCKING=1

def get_logger(file, level: str=logging.INFO):
    # initialize logger
    logging.basicConfig(level=level, filename=file, filemode="w+",
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logger = logging.getLogger(__name__)
    return logger

def set_seed(manual_seed=0):
    random.seed(manual_seed)
    np.random.seed(seed=manual_seed)
    set_determinism(seed=manual_seed)
    torch.manual_seed(seed=manual_seed)
    return

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    __spec__ = None # for pdb
    print_config()

    # parse cmd args to get config file
    parser = argparse.ArgumentParser(description='Command-line arguments for 3D UNet Training')
    parser.add_argument('-c', '--config', type=str, default='config.yml', required=False,
                        help='Path to the configuration file')
    parser.add_argument('-ca', '--contrast_agent', type=evaluate_true_false, default=False, required=True, 
                        help="Dataset contains contrast agent or not")
    args = parser.parse_args()

    # load configuration from file
    config = load_config(args.config)

    log_path = Path(os.getcwd() + '/output/')

    if config.get('train', True):
        if not os.path.exists(log_path.joinpath(config['logs'])):
            os.makedirs(log_path.joinpath(config['logs']))

        # setup logging
        logger = get_logger(log_path.joinpath(config['logs']).joinpath('training_logs.log'))
        logger.info('Starting training.')

        logger.info('Setting seed = {}'.format(0))
        if config['use_seed']: set_seed(manual_seed=config['seed'])

        # write hyperparams to config file
        with open(log_path.joinpath(config['logs']).joinpath('config.yml'), 'w') as config_file:
            yaml.dump(config, config_file)
        if args.contrast_agent:
            model = train_contrast(config, log_path, logger)
        elif not args.contrast_agent:
            model = train_non_contrast(config, log_path, logger)
    else:
        pass