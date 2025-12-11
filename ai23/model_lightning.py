import torch
import torch.nn.functional as F
from torch import nn, cat, optim
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
import utility


def kaiming_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super().__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernelx, stridex, paddingx)
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(self, channel, final=False):
        super().__init__()
        if final:
            self.conv_block0 = ConvBNRelu([channel[0], channel[0]])
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(channel[0], channel[1], kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.conv_block0 = ConvBNRelu([channel[0], channel[1]])
            self.conv_block1 = ConvBNRelu([channel[1], channel[1]])
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.conv_block0(x)
        return self.conv_block1(x)


class ai23(pl.LightningModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config.GlobalConfig
        self.gpu_device = device
        self.automatic_optimization = False

        # Normalizer and Encoders
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b3(pretrained=True)
        self.RGB_encoder.classifier = nn.Sequential()
        self.RGB_encoder.avgpool = nn.Sequential()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_ss_f = ConvBlock([self.config.n_fmap_b3[4][-1] + self.config.n_fmap_b3[3][-1],
                                     self.config.n_fmap_b3[3][-1]])
        self.conv2_ss_f = ConvBlock([self.config.n_fmap_b3[3][-1] + self.config.n_fmap_b3[2][-1],
                                     self.config.n_fmap_b3[2][-1]])
        self.conv1_ss_f = ConvBlock([self.config.n_fmap_b3[2][-1] + self.config.n_fmap_b3[1][-1],
                                     self.config.n_fmap_b3[1][-1]])
        self.conv0_ss_f = ConvBlock([self.config.n_fmap_b3[1][-1] + self.config.n_fmap_b3[0][-1],
                                     self.config.n_fmap_b3[0][0]])
        self.final_ss_f = ConvBlock([self.config.n_fmap_b3[0][0], self.config.n_class], final=True)

        # Semantic Cloud Encoder
        self.SC_encoder = models.efficientnet_b1(pretrained=False)
        self.SC_encoder.features[0][0] = nn.Conv2d(self.config.n_class,
                                                   self.config.n_fmap_b1[0][0],
                                                   kernel_size=3, stride=2, padding=1, bias=False)
        self.SC_encoder.classifier = nn.Sequential()
        self.SC_encoder.avgpool = nn.Sequential()
        self.SC_encoder.apply(kaiming_init)

        # Fusion + GRU + Waypoint
        self.necks_net = nn.Sequential(
            nn.Conv2d(self.config.n_fmap_b3[4][-1] + self.config.n_fmap_b1[4][-1],
                      self.config.n_fmap_b3[4][1], 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.config.n_fmap_b3[4][1], self.config.n_fmap_b3[4][0])
        )
        self.gru = nn.GRUCell(input_size=7, hidden_size=self.config.n_fmap_b3[4][0])
        self.pred_dwp = nn.Linear(self.config.n_fmap_b3[4][0], 2)

    # ===================================================================

    def forward(self, rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, velo_in):
        RGB_features_sum, SC_features_sum = 0, 0
        segs_f, sdcs = [], []

        for i in range(self.config.seq_len):
            in_rgb = self.rgb_normalizer(rgbs[i])
            feats = [in_rgb]
            for block in self.RGB_encoder.features:
                feats.append(block(feats[-1]))
            RGB_features_sum += feats[-1]

            ss_f_3 = self.conv3_ss_f(cat([self.up(feats[-1]), feats[5]], dim=1))
            ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), feats[3]], dim=1))
            ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), feats[2]], dim=1))
            ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), feats[1]], dim=1))
            ss_f = self.final_ss_f(self.up(ss_f_0))
            segs_f.append(ss_f)

            top_view_sc = self.gen_top_view_sc_ptcloud(pt_cloud_xs[i], pt_cloud_zs[i], ss_f)
            sdcs.append(top_view_sc)
            sc_feats = [top_view_sc]
            for block in self.SC_encoder.features:
                sc_feats.append(block(sc_feats[-1]))
            SC_features_sum += sc_feats[-1]

        hx = self.necks_net(cat([RGB_features_sum, SC_features_sum], dim=1))
        xy = torch.zeros(size=(hx.shape[0], 2), device=self.gpu_device, dtype=hx.dtype)
        velo_in = torch.tanh(velo_in.unsqueeze(1))
        rp1, rp2 = torch.tanh(rp1), torch.tanh(rp2)

        out_wp = []
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1)
            hx = self.gru(ins, hx)
            hx = torch.clamp(hx, -10, 10)
            d_xy = self.pred_dwp(hx)
            xy = xy + d_xy
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1)
        return segs_f, pred_wp, sdcs

    # ===================================================================

    def gen_top_view_sc_ptcloud(self, pt_cloud_x, pt_cloud_z, semseg):
        _, label_img = torch.max(semseg, dim=1)
        cloud_data_n = torch.arange(semseg.shape[0], device=self.gpu_device).repeat_interleave(self.config.crop_roi[0] * self.config.crop_roi[1])
        cloud_data_x = torch.round((pt_cloud_x + self.config.coverage_area) * (self.config.crop_roi[1]-1) / (2*self.config.coverage_area)).ravel()
        cloud_data_z = torch.round((pt_cloud_z * (1 - self.config.crop_roi[0]) / self.config.coverage_area) + (self.config.crop_roi[0]-1)).ravel()

        mask = (cloud_data_x >= 0) & (cloud_data_x < self.config.crop_roi[1]) & (cloud_data_z >= 0) & (cloud_data_z < self.config.crop_roi[0])
        coorx = torch.stack([cloud_data_n[mask], label_img.ravel()[mask], cloud_data_z[mask], cloud_data_x[mask]]).long()

        top_view_sc = torch.zeros_like(semseg)
        if coorx.shape[1] > 0:
            top_view_sc[coorx[0], coorx[1], coorx[2], coorx[3]] = 1.0
        return top_view_sc

    # ===================================================================

    def training_step(self, batch, batch_idx):
        opt, opt_lw = self.optimizers()
        config = self.config

        rgbs, segs, pt_cloud_xs, pt_cloud_zs = [], [], [], []
        for i in range(config.seq_len):
            rgbs.append(batch['rgbs'][i].float().to(self.device))
            segs.append(batch['segs'][i].float().to(self.device))
            pt_cloud_xs.append(batch['pt_cloud_xs'][i].float().to(self.device))
            pt_cloud_zs.append(batch['pt_cloud_zs'][i].float().to(self.device))

        rp1 = torch.stack(batch['rp1'], dim=1).float().to(self.device)
        rp2 = torch.stack(batch['rp2'], dim=1).float().to(self.device)
        gt_velocity = batch['velocity'].float().to(self.device)
        gt_waypoints = torch.stack(
            [torch.stack(batch['waypoints'][j], dim=1).float().to(self.device)
             for j in range(config.pred_len)], dim=1)

        pred_segs, pred_wp, _ = self(rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, gt_velocity)

        # Loss computation
        loss_seg = sum(utility.BCEDice(pred_segs[i], segs[i]) for i in range(config.seq_len)) / config.seq_len
        loss_wp = F.l1_loss(pred_wp, gt_waypoints)

        params_lw = opt_lw.param_groups[0]['params']
        total_loss = params_lw[0]*loss_seg + params_lw[1]*loss_wp

        # --- GradNorm last batch logic ---
        opt.zero_grad(set_to_none=True)
        if batch_idx == 0:
            self.loss_seg_0, self.loss_wp_0 = loss_seg.detach(), loss_wp.detach()
            self.manual_backward(total_loss)
        elif (batch_idx + 1) < self.trainer.num_training_batches:
            self.manual_backward(total_loss)
        else:
            if config.MGN:
                opt_lw.zero_grad(set_to_none=True)
                self.manual_backward(total_loss, retain_graph=True)

                params = [p for p in self.parameters() if p.requires_grad]
                G0R = torch.autograd.grad(loss_seg, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
                G1R = torch.autograd.grad(loss_wp, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
                G0, G1 = torch.norm(G0R[0], keepdim=True), torch.norm(G1R[0], keepdim=True)
                G_avg = (G0 + G1) / 2

                loss_seg_hat = loss_seg / (self.loss_seg_0 + 1e-8)
                loss_wp_hat = loss_wp / (self.loss_wp_0 + 1e-8)
                loss_hat_avg = (loss_seg_hat + loss_wp_hat) / 2

                inv_rate_ss = loss_seg_hat / loss_hat_avg
                inv_rate_wp = loss_wp_hat / loss_hat_avg

                C0 = (G_avg * inv_rate_ss).detach() ** config.lw_alpha
                C1 = (G_avg * inv_rate_wp).detach() ** config.lw_alpha
                Lgrad = F.l1_loss(G0, C0) + F.l1_loss(G1, C1)

                self.manual_backward(Lgrad)
                opt_lw.step()
                self.lgrad = Lgrad.item()
            else:
                self.manual_backward(total_loss)
                self.lgrad = 0.0

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        opt.step()

        self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_seg_loss", loss_seg, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_wp_loss", loss_wp, prog_bar=False, on_step=True, on_epoch=True)
        return total_loss

    # ===================================================================

    def on_train_epoch_end(self):
        if not self.config.MGN:
            return
        opt_lw = self.optimizers()[1]
        params_lw = opt_lw.param_groups[0]['params']
        lw = np.array([p.detach().cpu().numpy()[0] for p in params_lw])
        coef = np.array(self.config.loss_weights).sum() / lw.sum()
        new_lw = [torch.tensor([coef * x], device='cuda', dtype=torch.float32, requires_grad=True) for x in lw]
        opt_lw.param_groups[0]['params'] = new_lw

    # ===================================================================

    def configure_optimizers(self):
        optima = optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.1, patience=4, min_lr=1e-6)
        params_lw = [torch.tensor([w], dtype=torch.float32, device='cuda').requires_grad_(True)
                     for w in self.config.loss_weights]
        optima_lw = optim.SGD(params_lw, lr=self.config.lr)
        return [
            {"optimizer": optima, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_total_loss"}},
            {"optimizer": optima_lw},
        ]
