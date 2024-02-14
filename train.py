import torch
import os
import random
import traceback
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from monai.data import DataLoader, CacheDataset, Dataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.data.utils import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, UNETR
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import (ToTensord, Compose, LoadImaged, ToTensord, Spacingd, Transposed, Flipd, EnsureChannelFirstd, 
                              EnsureType, Compose, AsDiscrete, RandSpatialCropSamplesd, RandFlipd, RandShiftIntensityd, SpatialPadd,
                              RandGaussianNoised, ThresholdIntensityd)
from utils import *
from pathlib import Path


# from tqdm import tqdm # can be added if not running in the background

def save_checkpoint(model_state_dict, 
                    optimizer_seg_state_dict, 
                    save_path=None):
    """Save checkpoint while training the model

    Args:
        model_state_dict (dict): Dictionary containing model state i.e. weights and biases
            Required: True
        optimizer_state_dict (dict): Dictionary containing optimizer state for the segmentation part i.e. gradients
            Required: True
        save_path (str): Path to save the checkpoint
            Required: False     Default: None  
    Returns:
        -
    """
    torch.save({'model_state_dict': model_state_dict,
                'optimizer_seg_state_dict': optimizer_seg_state_dict,
                }, save_path)
    
def key_error_raiser(ex): raise Exception(ex)

def train(config, log_path, logger):

    cases = [f for f in Path(config['data_dir']).glob('*')]
    image_files = sorted([os.path.join(k, f) for k in cases for f in k.glob('*') if f.is_dir()])
    right_seg_files = sorted([os.path.join(k, f) for k in cases for f in k.glob('*') if 'right' in str(f)])
    left_seg_files = sorted([os.path.join(k, f) for k in cases for f in k.glob('*') if 'left' in str(f)])

    train_transforms_config = config['train_transforms']
    eval_transforms_config = config['eval_transforms']
    train_transforms = Compose([
                            LoadImaged(keys=train_transforms_config['LoadImaged_im']['keys'], 
                                       ensure_channel_first=train_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            LoadImaged(keys=train_transforms_config['LoadImaged_seg']['keys'], 
                                       ensure_channel_first=train_transforms_config['LoadImaged_seg']['ensure_channel_first']),
                            # Spacingd(keys=train_transforms_config['Spacingd_im']['keys'], 
                            #          pixdim=train_transforms_config['Spacingd_im']['pixdim']),
                            # Spacingd(keys=train_transforms_config['Spacingd_seg']['keys'], 
                            #          pixdim=train_transforms_config['Spacingd_seg']['pixdim'], mode=train_transforms_config['Spacingd_seg']['mode']),
                            Transposed(keys=train_transforms_config['Transposed']['keys'], 
                                       indices=train_transforms_config['Transposed']['indices']),
                            WindowindContrastCTd(keys=train_transforms_config['WindowindContrastCTd']['keys']),
                            ConvertToMultiChannelMaskd(keys=train_transforms_config['ConvertToMultiChannelMaskd']['keys']), 
                            SpatialPadd(keys=train_transforms_config['SpatialPadd']['keys'], 
                                        spatial_size=train_transforms_config['SpatialPadd']['spatial_size']),
                            # RandSpatialCropSamplesd(keys=train_transforms_config['RandSpatialCropSamplesd']['keys'], 
                            #                         roi_size=train_transforms_config['RandSpatialCropSamplesd']['roi_size'], 
                            #                         num_samples=train_transforms_config['RandSpatialCropSamplesd']['num_samples'], 
                            #                         random_size=train_transforms_config['RandSpatialCropSamplesd']['random_size']),
                            # RandFlipd(keys=train_transforms_config['RandFlipd_x']['keys'], 
                            #           prob=train_transforms_config['RandFlipd_x']['prob'], 
                            #           spatial_axis=train_transforms_config['RandFlipd_x']['spatial_axis']), 
                            # RandFlipd(keys=train_transforms_config['RandFlipd_y']['keys'], 
                            #           prob=train_transforms_config['RandFlipd_y']['prob'], 
                            #           spatial_axis=train_transforms_config['RandFlipd_y']['spatial_axis']), 
                            # RandFlipd(keys=train_transforms_config['RandFlipd_z']['keys'], 
                            #           prob=train_transforms_config['RandFlipd_z']['prob'], 
                            #           spatial_axis=train_transforms_config['RandFlipd_z']['spatial_axis']), 
                            # RandShiftIntensityd(keys=train_transforms_config['RandShiftIntensityd']['keys'], 
                            #                     offsets=train_transforms_config['RandShiftIntensityd']['offsets'], 
                            #                     prob=train_transforms_config['RandShiftIntensityd']['prob']),
                            # RandGaussianNoised(keys=train_transforms_config['RandGaussianNoised']['keys'], 
                            #                    prob=train_transforms_config['RandGaussianNoised']['prob'], 
                            #                    mean=train_transforms_config['RandGaussianNoised']['mean'], 
                            #                    std=train_transforms_config['RandGaussianNoised']['std']), 
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_upper']['keys'], 
                            #                     threshold=train_transforms_config['ThresholdIntensityd_clip_upper']['threshold'], 
                            #                     above=train_transforms_config['ThresholdIntensityd_clip_upper']['above'], 
                            #                     cval=train_transforms_config['ThresholdIntensityd_clip_upper']['cval']),
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_lower']['keys'], 
                            #                     threshold=train_transforms_config['ThresholdIntensityd_clip_lower']['threshold'], 
                            #                     above=train_transforms_config['ThresholdIntensityd_clip_lower']['above'], 
                            #                     cval=train_transforms_config['ThresholdIntensityd_clip_lower']['cval']),
                            ToTensord(keys=train_transforms_config['ToTensord']['keys'])
                        ])

    val_transforms = Compose([
                            LoadImaged(keys=eval_transforms_config['LoadImaged_im']['keys'], 
                                       ensure_channel_first=eval_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            LoadImaged(keys=eval_transforms_config['LoadImaged_seg']['keys'], 
                                       ensure_channel_first=eval_transforms_config['LoadImaged_seg']['ensure_channel_first']), 
                            # Spacingd(keys=eval_transforms_config['Spacingd_im']['keys'], 
                            #          pixdim=eval_transforms_config['Spacingd_im']['pixdim']),
                            # Spacingd(keys=eval_transforms_config['Spacingd_seg']['keys'], 
                            #          pixdim=eval_transforms_config['Spacingd_seg']['pixdim'], mode=eval_transforms_config['Spacingd_seg']['mode']),
                            Transposed(keys=eval_transforms_config['Transposed']['keys'], 
                                       indices=eval_transforms_config['Transposed']['indices']),
                            WindowindContrastCTd(keys=eval_transforms_config['WindowindContrastCTd']['keys']), 
                            ConvertToMultiChannelMaskd(keys=eval_transforms_config['ConvertToMultiChannelMaskd']['keys']),
                            SpatialPadd(keys=eval_transforms_config['SpatialPadd']['keys'], 
                                        spatial_size=eval_transforms_config['SpatialPadd']['spatial_size']),
                            ToTensord(keys=eval_transforms_config['ToTensord']['keys'])
                        ])

    datadict = [{"image": im, "right_seg": right_seg, "left_seg": left_seg} 
                    for im, right_seg, left_seg in zip(image_files, right_seg_files, left_seg_files)]
    
    cross_val_split = config['cross_val_split'] if 'cross_val_split' in config.keys() else key_error_raiser("Cross validation split not defined in config.")
    
    random.shuffle(datadict)
    train_dict = datadict[:int(len(datadict) * cross_val_split)]
    val_dict = datadict[int(len(datadict) * cross_val_split):]
    train_dict = datadict 
    val_dict = datadict 
    logger.info('Dataset length {} '. format(len(datadict)))
    logger.info('Train/Val split {} , {}'. format(train_dict, val_dict))

    # define dataset
    # train_dataset = CacheDataset(data=train_dict, transform=train_transforms)
    # val_dataset = CacheDataset(data=train_dict, transform=val_transforms)
    train_dataset = Dataset(data=train_dict, transform=train_transforms)
    val_dataset = Dataset(data=val_dict, transform=val_transforms)

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    # initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'] , 
                              shuffle=config['dataloader']['shuffle'], 
                              num_workers=config['dataloader']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=1, 
                            shuffle=config['dataloader']['shuffle'],
                            num_workers=config['dataloader']['num_workers'])

    # while 1:
    #     for i, sample in enumerate(train_loader):
    #         print(i)
    #         # # How to user IndexTracker
    #         fig, ax = plt.subplots(1, 1)
    #         X = np.hstack((sample['image'][0,0].detach().cpu().numpy(), sample['segmentation'][0,0].detach().cpu().numpy(),
    #                         sample['segmentation'][0,1].detach().cpu().numpy(), sample['segmentation'][0,2].detach().cpu().numpy()))
    #         tracker = IndexTracker(ax, X, vmin=np.amin(X), vmax=np.amax(X))
    #         fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    #         plt.show()
    # exit(0)
    while 1:
        for i, sample in enumerate(train_loader):
            print(i)
            # # How to user IndexTracker
            fig, ax = plt.subplots(1, 1)
            X = np.hstack((sample['image'][0,0].detach().cpu().numpy(), sample['segmentation'][0,1].detach().cpu().numpy()+sample['segmentation'][0,2].detach().cpu().numpy()))
            tracker = IndexTracker(ax, X, vmin=np.amin(X), vmax=np.amax(X))
            fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
            plt.show()
    exit(0)


    # initialize model
    if config['model']['name'] == 'UNETR': 
        model = UNETR(in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'], img_size=config['model']['img_size'], 
                      feature_size=config['model']['feature_size'], hidden_size=config['model']['hidden_size'], mlp_dim=config['model']['mlp_dim'], 
                      num_heads=config['model']['num_heads'], pos_embed=config['model']['pos_embed'], norm_name=config['model']['norm_name'], 
                      conv_block=config['model']['conv_block'], res_block=config['model']['res_block'], dropout_rate=config['model']['dropout_rate'])
    elif config['model']['name'] == 'UNet':
        model = UNet(spatial_dims=config['model']['spatial_dims'], in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'],
                     kernel_size=config['model']['kernel_size'], up_kernel_size=config['model']['up_kernel_size'], channels=config['model']['channels'],
                     strides=config['model']['strides'], norm=config['model']['norm'], dropout=config['model']['dropout'], 
                     num_res_units=config['model']['num_res_units'])
    else: 
        raise Exception("No model has been defined in the config file")
    
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model) # use multiple GPUs
    model.to(device=torch.device(config['device']))
    
    # initialize optimizer
    if 'optimizer' in config.keys():
        if config['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['learning_rate'], betas=config['optimizer']['betas'], 
                                        weight_decay=config['optimizer']['weight_decay']) 
    else: 
        raise Exception("No optimizer has been defined in the config file")
    logger.info('Training with optimizer {} '.format(optimizer))

    # initialize scheduler
    if 'scheduler' in config.keys():
        if config['scheduler']['name'] == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=config['scheduler']['T_0'], eta_min=config['scheduler']['eta_min'])
            logger.info('Scheduler {}'.format(scheduler))
    else:
        logger.info('No scheduler for the learning rate has been defined')

    # initialize loss
    if 'loss' in config.keys():
        if config['loss']['name'] == 'DiceLoss':
            loss = DiceLoss(softmax=config['loss']['softmax'], include_background=config['loss']['include_background'])
        elif config['loss']['name'] == 'DiceCELoss':
            loss = DiceCELoss(softmax=config['loss']['softmax'], include_background=config['loss']['include_background'])
    else:
        raise Exception("No loss has been defined in the config file.")
    logger.info('Loss function to minimize {}.'.format(loss))

    post_pred_transforms = config['post_pred_transforms'] if 'post_pred_transforms' in config.keys() else key_error_raiser("Post-prediction transforms not defined config.")
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=post_pred_transforms['AsDiscrete']['argmax'], 
                                                  to_onehot=post_pred_transforms['AsDiscrete']['to_onehot'])])
    post_label = Compose([EnsureType()])

    if 'metric' in config.keys():
        if config['metric']['name'] == 'DiceMetric':
            train_metric = DiceMetric(include_background=config['metric']['include_background'], reduction=config['metric']['reduction'])
            val_metric =  DiceMetric(include_background=config['metric']['include_background'], reduction=config['metric']['reduction'])
    else:
        raise Exception("No metric has been defined in the config file")
    logger.info('Metric {}'.format(train_metric))

    num_epochs = config['epochs'] if 'epochs' in config.keys() else key_error_raiser("Number of epochs missing from config.")
    val_interval = config['val_interval'] if 'val_interval' in config.keys() else key_error_raiser("Validation interval not defined in config.")
    log_file = config['logs'] if 'logs' in config.keys() else key_error_raiser("Log file not defined in config.")
    device = config['device'] if 'device' in config.keys() else key_error_raiser("Device not defined in config.")
    writer = SummaryWriter()
    losses = []
    val_losses = []
    validation_metrics = []

    for epoch in range(config['epochs']):
        model.train()

        for batch, train_data in enumerate(train_loader, 1):
            image, segmentation = train_data['image'].float().to(device=torch.device(config['device'])), train_data['segmentation'].float().to(device=torch.device(config['device']))

            try:
                optimizer.zero_grad()
                out = model(image)

                loss_s = loss(out, segmentation)
                loss_s.backward()

                _outputs = [post_pred(i) for i in decollate_batch(out)]
                _labels = [post_label(i) for i in decollate_batch(segmentation)]

                optimizer.step()
                train_metric(y_pred=_outputs, y=_labels)

            except Exception as e:
                print('Caught the following exception {}'.format(traceback.format_exc()))
            losses.append(loss_s.item())
        metric = train_metric.aggregate().item()
        'scheduler' in config.keys() and scheduler.step()

        # if metric > 0.5:
        plot_2d_or_3d_image(data=image, step=0, writer=writer, frame_dim=-1, tag=f'image at epoch: {epoch}')
        plot_2d_or_3d_image(data=segmentation, step=0, writer=writer, frame_dim=-1, tag=f'label at epoch: {epoch}')
        plot_2d_or_3d_image(data=out, step=0, writer=writer, frame_dim=-1, tag=f'model output at epoch: {epoch}')

        writer.add_scalar(tag='Loss/train', scalar_value=losses[-1], global_step=epoch)
        logger.info(f'Epoch {epoch} of {config["epochs"]} with Train loss {losses[-1]}')
        logger.info(f'Epoch {epoch} of {config["epochs"]} with Train metric {metric}')
        logger.info(f'-------------- Finished epoch {epoch} -------------')
        train_metric.reset()

        if epoch % config['val_interval'] == 0:
            with torch.no_grad():
                # evaluate model
                model.eval()

                for _, val_data in enumerate(val_loader, 1):
                    val_image, val_segm = val_data['image'].float().to(device=torch.device(config['device'])), val_data['segmentation'].float().to(device=torch.device(config['device']))

                    try:
                        val_out = sliding_window_inference(inputs=val_image, roi_size=(96, 96, 96), sw_batch_size=16, predictor=model)

                        loss_s = loss(val_out, val_segm)

                        val_outputs = [post_pred(i) for i in decollate_batch(val_out)]
                        val_labels = [post_label(i) for i in decollate_batch(val_segm)]

                        val_metric(val_outputs, val_labels)
                    except Exception as e:
                        print(f'Exception caught while validating in {traceback.format_exc()}. Aborting...')
                    # record loss
                    val_losses.append(loss_s.item())
                metric = val_metric.aggregate().item()
                validation_metrics.append(metric)

                writer.add_scalar(tag='Loss/eval', scalar_value=val_losses[-1], global_step=epoch)
                logger.info(f'Eval loss {val_losses[-1]}')
                logger.info(f'Eval metric {metric}')
                logger.info(f'-------------- Finished epoch {epoch} -------------') 
                val_metric.reset()

                # save models
                if validation_metrics[-1] == max(validation_metrics):
                    if not os.path.exists(log_path.joinpath(config['logs']).joinpath('models')):
                        os.makedirs(log_path.joinpath(config['logs']).joinpath('models'))
                    logger.info(f'Saving model at epoch {epoch}')
                    save_checkpoint(model_state_dict=model.state_dict(), optimizer_seg_state_dict=optimizer.state_dict(), 
                                    save_path=log_path.joinpath(config['logs']).joinpath('models/model.tar'.format(epoch)))

    return model  