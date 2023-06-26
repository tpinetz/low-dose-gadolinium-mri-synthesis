import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid

from loss_criterions import get_loss_criterion, VGGPerceptualLoss, psnr_criterion, PatchWiseWasserStein, PatchWiseWasserSteinSinkhorn
import networks
import networks_cond
import discriminator
from torch_utils import get_gaussian_filter_3d


class GanModel(pl.LightningModule):
    def __init__(self, cfg, fast: bool = False):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.generator = networks_cond.CondMultiScaleModel(config=self.cfg.generator, fast=fast)
        self.discriminator = discriminator.Discriminator(config=self.cfg.model, fast=fast)

        # define the loss criterion
        self.loss_criterion = get_loss_criterion(cfg.train.get("criterion", "l1"))
        gauss_filter_torch = {key.value: get_gaussian_filter_3d(1/s) for key, s in zip(networks.Resolution,
                                                                                       self.cfg.train.spacings)}
        self.gauss_filter = get_gaussian_filter_3d(1.)
        self.gauss_filter.stride = (2, 2, 2)
        self.vgg_loss = VGGPerceptualLoss()
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_loss": 0, "hp/val_psnr": 0})

    def forward(self, data, z):
        return self.generator(data, z)

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.optim.gen_learning_rate,
                                     betas=(0.5, 0.9))
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.optim.disc_learning_rate,
                                     betas=(0.5, 0.9))

        return ({'optimizer': gen_optimizer, 'frequency': 1},
                {'optimizer': disc_optimizer, 'frequency': 5})

    def training_step(self, batch, batch_idx, optimizer_idx):
        sample = batch

        x = sample['data']
        real_cond = sample['condition']
        l_mask_atlas = sample['mask_atlas']
        x *= l_mask_atlas
        condition = real_cond.clone()
        condition[:,0] = 0.05 + torch.rand(condition[:,0].shape).type_as(condition) * 0.45
        target = sample['target'] * l_mask_atlas


        if optimizer_idx == 0:
            noise_real = torch.normal(0, 1, size=target.shape).type_as(target)
            noise_gen = torch.normal(0, 1, size=target.shape).type_as(target)
            # other_noise = torch.normal(0, 1, size=target.shape).type_as(target)
            transformed_imgs = self(torch.cat([noise_real, x], 1), real_cond)
            generated_imgs = self(torch.cat([noise_gen, x], 1), condition)
            # other_generated_imgs = self(torch.cat([other_noise, x], 1), condition)

            mask_atlas = [l_mask_atlas]
            for i in range(len(generated_imgs) - 1):
                mask_atlas.insert(0, self.generator.down(mask_atlas[0]))

            # noise_loss = sum([torch.mean(mask_atlas[i] * torch.abs(generated_imgs[i] - other_generated_imgs[i])) for i in range(len(generated_imgs))])

            targets = [target]
            for i in range(2):
                targets.insert(0, self.gauss_filter(targets[0]))

            _, _, z, _, _ = transformed_imgs[-1].shape
            l1_loss = 0
            if self.cfg.train.use_vgg:
                for i in range(1, z - 2):
                    l1_loss += torch.mean(self.vgg_loss(transformed_imgs[-1][:, 0, i - 1:i + 2],
                                                        targets[-1][:, 0, i - 1:i + 2]))
            else:
                l1_loss = sum([torch.abs(transformed_imgs[i] - targets[i]).mean() for i in range(len(transformed_imgs))])
            gen_loss = torch.mean(self.discriminator(torch.cat([generated_imgs[-1], x], 1), condition))
            loss =  self.cfg.train.w0 * l1_loss +  self.cfg.train.w1 * gen_loss # - self.cfg.train.noise_loss * noise_loss

            if self.cfg.train.w2 > 0:
                wdist_noise_loss = 0
                for i in range(len(targets)):
                    if self.cfg.train.sinkhorn:
                        wdist_noise_loss += PatchWiseWasserSteinSinkhorn(transformed_imgs[i], targets[i], 4, 4)
                    else:
                        wdist_noise_loss += PatchWiseWasserStein(transformed_imgs[i], targets[i], 3, 2)
                loss += self.cfg.train.w2 * wdist_noise_loss
                self.log('loss/wdist_noise_loss', wdist_noise_loss)

            self.log('loss/l1_generator', l1_loss)
            self.log('loss/gen_disc_loss', gen_loss)
            self.log('loss/generator', loss)
            # self.log('loss/noise_loss', noise_loss)

            if self.global_step % self.cfg.train.num_log == 0 or self.global_step == self.cfg.optim.num_iter - 1:
                # training set visualization
                # self.logger.experiment.add_image(f'train/0-T1_sub', make_grid(x[:16, 0:1, 10, :, :], nrow=4, normalize=True), self.global_step)
                self.logger.experiment.add_image(f'train/1-T1_nativ', make_grid(x[:16, 0:1, 10, :, :], nrow=4, normalize=True), self.global_step)
                self.logger.experiment.add_image(f'train/2-T1_full', make_grid(x[:16, 1:2, 10, :, :], nrow=4, normalize=True), self.global_step)
                self.logger.experiment.add_image(f'train/3-pred', make_grid(transformed_imgs[-1][:16, :, 10, :, :], nrow=4, normalize=True), self.global_step)
                self.logger.experiment.add_image(f'train/3-pred_random', make_grid(generated_imgs[-1][:16, :1, 10, :, :], nrow=4, normalize=True), self.global_step)
                self.logger.experiment.add_image(f'train/8-target', make_grid(target[:16, :, 10, :, :], nrow=4, normalize=True), self.global_step)

        if optimizer_idx == 1:
            noise = torch.normal(0, 1, size=target.shape).type_as(target)
            gen_imgs = self(torch.cat([noise, x], 1), condition)[-1].detach()

            real_cond_smoothed = real_cond + torch.normal(0, 0.05, size=real_cond.shape).type_as(real_cond)
            alpha = torch.rand(len(gen_imgs), 1, 1, 1, 1).type_as(gen_imgs)
            interpolates = alpha * target + (1 - alpha) * gen_imgs
            alpha = alpha.view(-1, 1)
            interpolated_cond = (alpha * real_cond_smoothed + (1 - alpha) * condition)

            interpolates = torch.cat([interpolates, x], dim=1)
            interpolates.requires_grad = True
            interpolated_cond.requires_grad = True
            prop_interpolates = self.discriminator(interpolates, interpolated_cond)
            gradients = torch.autograd.grad(outputs=prop_interpolates.mean(), inputs=[interpolates, interpolated_cond], 
                                            create_graph=True, retain_graph=True)
            # Only use the gradient with respect to the generated image.
            grad_penalty_imgs = self.cfg.train.lambda_gp * ((1 - torch.sqrt(torch.sum(gradients[0].view(len(gen_imgs), -1) ** 2, -1) + torch.sum(gradients[1].view(len(gen_imgs), -1) ** 2, -1) + 1e-6)) ** 2).mean()

            wdist_estimate = torch.mean(self.discriminator(torch.cat([target, x], dim=1), real_cond_smoothed) -
                                        self.discriminator(torch.cat([gen_imgs, x], dim=1), condition))
            loss =  wdist_estimate + grad_penalty_imgs

            self.log('loss/discriminator_wdist', -wdist_estimate)
            self.log('loss/discriminator', loss)
            self.log('loss/gd_pen_imgs', grad_penalty_imgs)
            # self.log('loss/gd_pen_cond', grad_penalty_cond)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        sample = batch
        spacing = sample['resolution'].item()
        x = sample['data']
        l_mask_atlas = sample['mask_atlas']
        x *= l_mask_atlas
        condition = sample['condition']
        target = sample['target'] * l_mask_atlas
        noise = torch.normal(0, 1, size=target.shape).type_as(target)
        generated_imgs = self(torch.cat([noise, x], 1), condition)[-1]
        l1_loss = torch.abs(target - generated_imgs)

        if batch_idx < 4:
            s = 100
            if self.global_step == 0:
                # self.logger.experiment.add_image(f'val-{batch_idx}-{spacing}/0-T1_sub', make_grid(x[:, 0:1, s, :, :], nrow=4, normalize=True), global_step=self.global_step)
                self.logger.experiment.add_image(f'val-{batch_idx}-{spacing}/1-T1_nativ', make_grid(x[:, 0:1, s, :, :], nrow=4, normalize=True), global_step=self.global_step)
                self.logger.experiment.add_image(f'val-{batch_idx}-{spacing}/2-T1_full', make_grid(x[:, 1:2, s, :, :], nrow=4, normalize=True), global_step=self.global_step)
                self.logger.experiment.add_image(f'val-{batch_idx}-{spacing}/3-loss', make_grid(l1_loss[:, :, s, :, :], nrow=4, normalize=True), global_step=self.global_step)
                self.logger.experiment.add_image(f'val-{batch_idx}-{spacing}/8-target', make_grid(target[:, :, s, :, :], nrow=4, normalize=True), global_step=self.global_step)
            self.logger.experiment.add_image(f'val-{batch_idx}-{spacing}/7-pred', make_grid(generated_imgs[:, :, s, :, :], nrow=4, normalize=True), global_step=self.global_step)

        l1_loss = torch.mean(l1_loss)
        self.log('loss/val', l1_loss)

        return {'val_loss': l1_loss}
