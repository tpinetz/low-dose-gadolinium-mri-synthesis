import click
import gan_model
import logging
import numpy as np
import os
import torch
import re

from pathlib import Path

ch = logging.StreamHandler()

logging.basicConfig(level=logging.INFO,
                    handlers=[
                        ch,
                    ])
from registration import radiometric_registration
import nibabel as nib


@click.command()
@click.pass_context
@click.option('--data_path', default='../data', help='Data for standard dose prediction', required=True)
@click.option('--ckpt', default='../checkpoints/model.ckpt', help='Checkpoint used for the deep learning model', required=True)
@click.option('--save_path', default='../results', help='Save path to store the result', required=True)
@click.option('--gpu', default=-1, type=int, help='Which gpu to run on. (-1 is cpu)', required=True)
@click.option('--ds', 'dose_level', type=float, help='Dosage level of the resulting scan', required=True)
@click.option('--fs', 'field_strength', default=3, type=int, help='Field strength of the input scan', required=True)
def main(ctx: click.Context,
         data_path: str,
         ckpt: str,
         save_path: str,
         gpu: int,
         dose_level: float,
         field_strength: float):
    checkpoint = torch.load(ckpt, map_location='cpu')
    checkpoint['hyper_parameters']['cfg']
    model = gan_model.GanModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model = model.eval()
    if gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')

    if dose_level is None:
        ctx.fail('--ds must be set to the correct dose level.')

    model = model.to(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_path = Path(data_path)
    for patient in data_path.iterdir():
        print(f"Computing patient: {patient}")
        t1_zero_nifti = nib.load([seg for seg in patient.glob("*") if re.match(".*t1?n.nii.gz", seg.name)][0])
        t1_zero = t1_zero_nifti.get_fdata()
        t1_full = nib.load(list(patient.glob("*t1c*"))[0]).get_fdata()

        vmax = np.quantile(t1_zero, 0.95)
        atlas_mask = t1_zero > 1e-6
        scale = radiometric_registration(t1_zero / vmax, t1_full / vmax, atlas_mask, max_iter = 1000)['params']

        nativ = t1_zero / vmax
        full = t1_full * scale / vmax

        with torch.no_grad():
            noise = np.random.normal(size=full.shape)
            inp = np.stack([noise, nativ, full])
            x = torch.from_numpy(inp[np.newaxis, ...].astype(np.float32)).to(device)
            cond = torch.from_numpy(np.array([[dose_level, (field_strength - 2.25) / 0.75]]).astype(np.float32)).to(device)
            prediction = model(x, cond)[-1]

            result_nifti = nib.Nifti1Image((nativ + prediction[0,0].detach().cpu().numpy()) * vmax, t1_zero_nifti.affine)
            nib.save(result_nifti, os.path.join(save_path, f"{patient.name}_prediction_{dose_level}.nii.gz"))



if __name__ == '__main__':
    main()