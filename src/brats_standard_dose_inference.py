import click
import logging
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import rotate
import torch


def psnr_criterion_np(x, y, mask=None):
    """
        Same behavior as psnr_criterion, just in numpy to evalute batches of volumes
        of the form: NxCxHxW.
        returns:
            psnr: np.float32
    """
    diff = x-y
    diff = diff.reshape((diff.shape[0], -1))
    y = y.reshape((y.shape[0], -1))
    if mask is None:
        v_max = np.max(y, 1)[0]
        N = diff.shape[1]
    else:
        mask = mask.reshape((mask.shape[0], -1))
        diff *= mask
        v_max = np.max(mask*y, axis=1)[0]
        N = np.sum(mask, axis=1)  # account for empty masks

    psnr = 20*np.log10(v_max / np.sqrt(np.sum(diff**2, axis=1)/N))
    # sort out infinite values
    return np.mean(psnr[np.isfinite(psnr)])


def normalize(x, vmin, vmax):
    return ((x - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def to_shape(x, shape):
    h = (x.shape[0] - shape[0]) // 2
    w = (x.shape[1] - shape[1]) // 2
    d = (x.shape[2] - shape[2]) // 2

    return x[h:h + shape[0], w: w+ shape[1], d:d + shape[2]]


def fill_data(data, patch_size):
    z, h, w = data.shape

    diffh = patch_size - h
    diffw = patch_size - w
    padh = 0
    padw = 0
    if diffh < 0:
        data[:, -diffh//2:diffh//2 + diffh % 2]
        diffh = 0
    else:
        padh = diffh // 2
    if diffw < 0:
        data[:, :, -diffw//2:diffw//2 + diffw % 2]
        diffw = 0
    else:
        padw = diffw // 2

    data = np.pad(data, ((0, 0), (padh, padh + diffh % 2), (padw, padw + diffw % 2)))

    return data.astype(np.float32)


def inference(sequences, network, device):
    predictions = []
    logging.info("Computing prediction:")
    for _a, axis in enumerate(range(3)):
        new_axis = [axis, ] + [_a for _a in range(3) if not _a == axis]
        filled_sequences = [fill_data(seq.transpose(new_axis), 256) for seq in sequences]
        prediction = np.zeros_like(filled_sequences[0])

        for i, angle in enumerate([0, 18, 36, 54, 72]):
            rot_sequences = [rotate(seq, angle, axes=(1, 2), reshape=False) for seq in filled_sequences]

            tmp_prediction = np.zeros_like(prediction)
            for slidx in range(3, rot_sequences[0].shape[0] - 3):
                sample = {
                    'data': np.concatenate([seq[slidx-3:slidx+4] for seq in rot_sequences], axis=0)[np.newaxis, ...].astype(np.float32)
                }
                x = torch.from_numpy(sample["data"]).to(device)
                with torch.no_grad():
                    diff = network(x)
                tmp_prediction[slidx] = diff[0,0].detach().cpu().numpy()
            print(i)
            prediction += rotate(tmp_prediction, -angle, axes=(1, 2), reshape=False)
            old_axis = [0, 0, 0]
            for i in range(3):
                old_axis[new_axis[i]] = i
        predictions.append(prediction.transpose(old_axis))

    prediction = predictions[0]
    h = (256 - predictions[0].shape[0]) // 2
    prediction += fill_data(predictions[1], 256)[h:h + predictions[0].shape[0]]
    prediction += fill_data(predictions[2], 256)[h:h + predictions[0].shape[0]]
    prediction /= 15

    # rescale to original range
    return prediction



@click.command()
@click.pass_context
@click.option('--data_path', default='../standard_dose_data', help='Data for standard dose prediction', required=True)
@click.option('--ckpt', default='../checkpoints/model_standard_dose_30p.ckpt', help='Checkpoint used for the deep learning model', required=True)
@click.option('--save_path', default='../results', help='Save path to store the result', required=True)
@click.option('--gpu', default=-1, type=int, help='Which gpu to run on. (-1 is cpu)', required=True)
def main(ctx: click.Context,
         data_path: str,
         ckpt: str,
         save_path: str,
         gpu: int):
    for patient in os.listdir(data_path):
        nativ_path = os.path.join(data_path, patient, patient + "_t1.nii.gz")
        low_path = os.path.join(data_path, patient, patient + "_t1cel.nii.gz")
        checkpoint = torch.load(ckpt, map_location='cpu')

        # load original model
        model = checkpoint['model']
        model.fast = True

        if gpu == -1:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{gpu}')
        
        model = model.to(device)

        nii_zero = nib.load(nativ_path)
        np_zero_orig = nii_zero.get_fdata() 
        np_low_orig = nib.load(low_path).get_fdata() 

        vmax = np.quantile(np_zero_orig, 0.95)

        np_zero = np_zero_orig / vmax
        np_low = np_low_orig / vmax

        sequences = [np_zero, np_low]
        prediction = inference(sequences, model, device) 
        prediction = np.maximum(to_shape(prediction, np_zero.shape), 0.) * vmax
        
        result_nifti = nib.Nifti1Image(prediction, nii_zero.affine)
        nib.save(result_nifti, os.path.join(save_path, f"{patient}_t1ce_prediction.nii.gz"))

    return 0


if __name__ == '__main__':
    main()



