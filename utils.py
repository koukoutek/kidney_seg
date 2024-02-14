import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.transforms import MapTransform, Resize
import nibabel as nib
import open3d as o3d

from math import floor, ceil
from monai.networks.nets import UNet


class ConvertToMultiChannelMaskd(MapTransform):
    """
        Convert multi-label singe-channel mask to multiple-channel one-hot encoded 
    """
    def __call__(self, data):
        d = dict(data)
        right_kidney = d['right_seg'][0]
        left_kidney = d['left_seg'][0]
        background = np.ones((right_kidney.shape[0], right_kidney.shape[1], right_kidney.shape[2]))
        background = background - (right_kidney + left_kidney)
        del d['left_seg']
        del d['right_seg']
        d['segmentation'] = np.stack((background, right_kidney, left_kidney))

        return d
    

class ResizeMaskd(MapTransform):
    """
        Convert multi-label singe-channel mask to multiple-channel one-hot encoded 
    """
    def __call__(self, data):
        d = dict(data)
        shape = d['right_seg'].shape
        print(d['segmentation'].shape, shape)
        d['segmentation'] = Resize(spatial_size=shape, mode='nearest')(d['segmentation'])
        return d


class WindowindContrastCTd(MapTransform):
    """
        Convert multi-label singe-channel mask to multiple-channel one-hot encoded 
    """
    def __call__(self, data):
        d = dict(data)
        # windowing should be based on the energy level
        d['image'][ d['image'] < -100 ] = -100
        d['image'][ d['image'] > 500 ] = 500
        d['image'] = (d['image'] - np.amin(d['image']))/np.ptp(d['image'])
        d['image_shape'] = d['image'][0].shape

        return d
    

class WindowindNonContrastCTd(MapTransform):
    """
        Convert multi-label singe-channel mask to multiple-channel one-hot encoded 
    """
    def __call__(self, data):
        d = dict(data)
        # windowing should be based on the energy level
        d['image'][ d['image'] < -105 ] = -105
        d['image'][ d['image'] > 230 ] = 230
        d['image'] = (d['image'] - np.amin(d['image']))/np.ptp(d['image'])
        d['image_shape'] = d['image'][0].shape

        return d


class IndexTracker:
    def __init__(self, ax, X, vmin, vmax):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.vmin = vmin
        self.vmax = vmax

        self.im = ax.imshow(self.X[:, :, self.ind], vmax=self.vmax, vmin=self.vmin, cmap='gray') #cmap='gray',
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step)) # print step and direction
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def evaluate_true_false(inp):
    inp = str(inp).upper()
    if 'TRUE'.startswith(inp):
        return True
    elif 'FALSE'.startswith(inp):
        return False
    else:
        raise ValueError('Argument error. Expected bool type.')
    
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def interpolate_image(image_data, thickness, x_spacing, y_spacing, spacing):
    """
        Interpolates image_data arg to have spacing voxel dimensions, given the current ones i.e. thickness, x_spacing, y_spacing
    """
    # get pixel dim and compute new spatial dimensions
    x_resize = floor((image_data.shape[0] / (spacing[0] / x_spacing)))
    y_resize = floor((image_data.shape[1] / (spacing[1] / y_spacing)))
    z_resize = floor((image_data.shape[2] / (spacing[2] / thickness)))
    # create tensor
    tensor_i = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0).to(device='cuda')
    tensor_i = F.interpolate(tensor_i, size=(x_resize, y_resize, z_resize), mode='trilinear', align_corners=True)
    # get data back to cpu
    image_data = tensor_i.to(device="cpu").detach().cpu().numpy()
    image_data = np.squeeze(image_data)
    return image_data

def read_nifti_data(path, header=False, affine=False):
    image = nib.load(path)
    if header and affine:
        return np.array(image.get_fdata()), image.header, image.affine
    elif header:
        return np.array(image.get_fdata()), image.header
    elif affine:
        return np.array(image.get_fdata()), image.affine
    else:
        return np.array(image.get_fdata())

def save_nifti(image_data, pixel_spacing, thickness, path, affine):
    image = nib.Nifti1Image(image_data, affine=affine)
    if pixel_spacing is not None:
        image.header.set_zooms(tuple(pixel_spacing) + (thickness,))
    nib.save(image, path)

def read_model(model_path):
    saved_model = torch.load(model_path)

    # 1-channel input
    model = UNet(spatial_dims=3, in_channels=1, out_channels=3, kernel_size=3, up_kernel_size=3, channels=[32, 64, 128, 256, 512],
                        strides=[2, 2, 2, 2], norm='instance', dropout=.3, num_res_units=2)
    model_dict = saved_model['model_state_dict']

    new_dict = {}
    for k,v in model_dict.items():
        if str(k).startswith('module'): # module will be there in case of training in multiple GPUs
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    model.load_state_dict(new_dict)

    return model

def object_oriented_bounding_box(predicted_r, predicted_l, spacing=(1,1,1), interpolate=True):
    
    if interpolate:
        # interpolation to (1,1,1) is needed in order to measure the 3 longest orthogonal axes inside each kidney
        right = interpolate_image(image_data=predicted_r.astype(np.float64), thickness=spacing[2], 
                                    x_spacing=spacing[0], y_spacing=spacing[1], spacing=(1,1,1))
        right[right > 0.6] = 1
        right[right <= 0.6] = 0

        # interpolation to (1,1,1) is needed in order to measure the 3 longest orthogonal axes inside each kidney
        left = interpolate_image(image_data=predicted_l.astype(np.float64), thickness=spacing[2], 
                                    x_spacing=spacing[0], y_spacing=spacing[1], spacing=(1,1,1))
        left[left > 0.6] = 1
        left[left <= 0.6] = 0
    else:
        right = predicted_r
        left = predicted_l
    
    if np.count_nonzero(right) != 0 :
        right_indexes = np.argwhere(right)
        right_pcd = o3d.geometry.PointCloud()
        right_pcd.points = o3d.utility.Vector3dVector(right_indexes)
        right_obb = right_pcd.get_oriented_bounding_box()
    else:
        right_obb = None
    if np.count_nonzero(left) != 0 :
        left_indexes = np.argwhere(left)
        left_pcd = o3d.geometry.PointCloud()
        left_pcd.points = o3d.utility.Vector3dVector(left_indexes)
        left_obb = left_pcd.get_oriented_bounding_box()
    else:
        left_obb = None

    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window()
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # visualizer.destroy_window()

    return right_obb, left_obb

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)

    from https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d

    """
    prediction = torch.from_numpy(prediction.astype(np.float16))
    truth = torch.from_numpy(truth.astype(np.float16))

    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives