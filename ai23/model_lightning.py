import os
from collections import deque
import sys
import csv
import numpy as np
import tqdm
import torch
from torch import nn, cat
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from utility import *
import utility
import config



#FUNGSI INISIALISASI WEIGHTS MODEL
#baca https://pytorch.org/docs/stable/nn.init.html
#kaiming he
def kaiming_init_layer(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    # layer.bias.data.fill_(0.01)

def kaiming_init(m):
    # print(m)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)

class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
        #weights initialization
        # kaiming_w_init(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.relu(x)
        return y

class ConvBlock(nn.Module):
    def __init__(self, channel, final=False): #up,
        super(ConvBlock, self).__init__()
        #conv block
        if final:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[0]], stridex=1)
            self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1),
            nn.Sigmoid()
            )
        else:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], stridex=1)
            self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], stridex=1)
        #init
        self.conv_block0.apply(kaiming_init)
        self.conv_block1.apply(kaiming_init)

    def forward(self, x):
        #convolutional block
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y

class ai23(pl.LightningModule):
    def __init__(self, config, device):#n_fmap, n_class=[23,10], n_wp=5, in_channel_dim=[3,2], spatial_dim=[240, 320], gpu_device=None):
        super(ai23, self).__init__()
        self.config = config.GlobalConfig
        self.gpu_device = device
        self.automatic_optimization = False
        #------------------------------------------------------------------------------------------------
        #RGB, jika inputnya sequence, maka jumlah input channel juga harus menyesuaikan
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b3(pretrained=True) #efficientnet_b4 bisa diganti
        self.RGB_encoder.classifier = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.RGB_encoder.avgpool = nn.Sequential() # type: ignore # cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        #SS
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_ss_f = ConvBlock(channel=[self.config.n_fmap_b3[4][-1]+self.config.n_fmap_b3[3][-1], self.config.n_fmap_b3[3][-1]])#, up=True)
        self.conv2_ss_f = ConvBlock(channel=[self.config.n_fmap_b3[3][-1]+self.config.n_fmap_b3[2][-1], self.config.n_fmap_b3[2][-1]])#, up=True)
        self.conv1_ss_f = ConvBlock(channel=[self.config.n_fmap_b3[2][-1]+self.config.n_fmap_b3[1][-1], self.config.n_fmap_b3[1][-1]])#, up=True)
        self.conv0_ss_f = ConvBlock(channel=[self.config.n_fmap_b3[1][-1]+self.config.n_fmap_b3[0][-1], self.config.n_fmap_b3[0][0]])#, up=True)
        self.final_ss_f = ConvBlock(channel=[self.config.n_fmap_b3[0][0], self.config.n_class], final=True)#, up=False)

        #untuk semantic cloud generator
        self.cover_area = self.config.coverage_area
        self.n_class = self.config.n_class
        self.h, self.w = int(self.config.crop_roi[0]/self.config.scale), int(self.config.crop_roi[1]/self.config.scale)
        #SC
        self.SC_encoder = models.efficientnet_b1(pretrained=False) #efficientnet_b0
        self.SC_encoder.features[0][0] = nn.Conv2d(self.config.n_class, self.config.n_fmap_b1[0][0], kernel_size=3, stride=2, padding=1, bias=False) #ganti input channel conv pertamanya, buat SC cloud
        self.SC_encoder.classifier = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.SC_encoder.avgpool = nn.Sequential()
        self.SC_encoder.apply(kaiming_init)

        #feature fusion
        self.necks_net = nn.Sequential( #inputnya dari 2 bottleneck
            nn.Conv2d(self.config.n_fmap_b3[4][-1]+self.config.n_fmap_b1[4][-1], self.config.n_fmap_b3[4][1], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.config.n_fmap_b3[4][1], self.config.n_fmap_b3[4][0])
        )

        self.gru = nn.GRUCell(input_size=7, hidden_size=self.config.n_fmap_b3[4][0])
        self.pred_dwp = nn.Linear(self.config.n_fmap_b3[4][0], 2)

    def forward(self, rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, velo_in):
        #bagian downsampling
        RGB_features_sum = 0
        SC_features_sum = 0
        segs_f = []
        sdcs = []

        #loop semua input dalam buffer
        for i in range(self.config.seq_len):
            in_rgb = self.rgb_normalizer(rgbs[i]) #
            RGB_features0 = self.RGB_encoder.features[0](in_rgb)
            RGB_features1 = self.RGB_encoder.features[1](RGB_features0)
            RGB_features2 = self.RGB_encoder.features[2](RGB_features1)
            RGB_features3 = self.RGB_encoder.features[3](RGB_features2)
            RGB_features4 = self.RGB_encoder.features[4](RGB_features3)
            RGB_features5 = self.RGB_encoder.features[5](RGB_features4)
            RGB_features6 = self.RGB_encoder.features[6](RGB_features5)
            RGB_features7 = self.RGB_encoder.features[7](RGB_features6)
            RGB_features8 = self.RGB_encoder.features[8](RGB_features7)
            RGB_features_sum += RGB_features8
            #bagian upsampling
            ss_f_3 = self.conv3_ss_f(cat([self.up(RGB_features8), RGB_features5], dim=1))
            ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features3], dim=1))
            ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features2], dim=1))
            ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features1], dim=1))
            ss_f = self.final_ss_f(self.up(ss_f_0))
            segs_f.append(ss_f)
            #------------------------------------------------------------------------------------------------
            #buat semantic cloud
            top_view_sc = self.gen_top_view_sc_ptcloud(pt_cloud_xs[i], pt_cloud_zs[i], ss_f)
            sdcs.append(top_view_sc)
            #bagian downsampling
            SC_features0 = self.SC_encoder.features[0](top_view_sc)
            SC_features1 = self.SC_encoder.features[1](SC_features0)
            SC_features2 = self.SC_encoder.features[2](SC_features1)
            SC_features3 = self.SC_encoder.features[3](SC_features2)
            SC_features4 = self.SC_encoder.features[4](SC_features3)
            SC_features5 = self.SC_encoder.features[5](SC_features4)
            SC_features6 = self.SC_encoder.features[6](SC_features5)
            SC_features7 = self.SC_encoder.features[7](SC_features6)
            SC_features8 = self.SC_encoder.features[8](SC_features7)
            SC_features_sum += SC_features8

        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hx = self.necks_net(cat([RGB_features_sum, SC_features_sum], dim=1))
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(hx.shape[0], 2)).to(self.gpu_device, dtype=hx.dtype)

        velo_in = velo_in.unsqueeze(1)

        #predict delta wp
        out_wp = list()
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1)
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx)
            xy = xy + d_xy
            out_wp.append(xy)
            # if nwp == 1: #ambil hidden state ketika sampai pada wp ke 2, karena 3, 4, dan 5 sebenarnya tidak dipakai
            #     hx_mlp = torch.clone(hx)
        pred_wp = torch.stack(out_wp, dim=1)

        return segs_f, pred_wp, sdcs
    
    def write_csv(self, total, seg, wp, lgrad):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.global_step,
                int(self.current_epoch),
                float(total),
                float(seg),
                float(wp),
                float(lgrad)
            ])

    
    def gen_top_view_sc_ptcloud(self, pt_cloud_x, pt_cloud_z, semseg):
        #proses awal
        _, label_img = torch.max(semseg, dim=1) #pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(self.h*self.w)] for n in range(semseg.shape[0])])).to(self.gpu_device, dtype=semseg.dtype)

        #normalize ke frame
        cloud_data_x = torch.round((pt_cloud_x + self.cover_area) * (self.w-1) / (2*self.cover_area)).ravel()
        cloud_data_z = torch.round((pt_cloud_z * (1-self.h) / self.cover_area) + (self.h-1)).ravel()

        #cari index interest
        bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= self.w-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        idx_xz = bool_xz.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya

        N = min(cloud_data_n.numel(), 
        label_img.numel(),
        cloud_data_z.numel(),
        cloud_data_x.numel())

        coorx = torch.stack([
            cloud_data_n[:N],
            label_img.ravel()[:N],
            cloud_data_z[:N],
            cloud_data_x[:N]
        ])

        # coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        coor_clsn = torch.unique(coorx[:, idx_xz]).long() #tensor harus long supaya bisa digunakan sebagai index
        top_view_sc = torch.zeros_like(semseg) #ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
        try:
            top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0
        except IndexError:
            # print("Warning: coor_clsn is empty, skipping this frame")
            pass
        return top_view_sc

    def training_step(self, batch, batch_idx):
        device = self.device
        config = self.config

        # Get optimizers manually (since automatic optimization is disabled)
        opt, opt_lw = self.optimizers()

        # Initialize loss trackers
        total_loss_meter = AverageMeter()
        seg_loss_meter = AverageMeter()
        wp_loss_meter = AverageMeter()

        # --------------------------
        # 1. Prepare input tensors
        # --------------------------
        rgbs, segs, pt_cloud_xs, pt_cloud_zs = [], [], [], []

        for i in range(config.seq_len):
            rgbs.append(batch['rgbs'][i].to(device, dtype=torch.float))
            segs.append(batch['segs'][i].to(device, dtype=torch.float))
            pt_cloud_xs.append(batch['pt_cloud_xs'][i].to(device, dtype=torch.float))
            pt_cloud_zs.append(batch['pt_cloud_zs'][i].to(device, dtype=torch.float))

        rp1 = torch.stack(batch['rp1'], dim=1).to(device, dtype=torch.float)
        rp2 = torch.stack(batch['rp2'], dim=1).to(device, dtype=torch.float)
        gt_velocity = batch['velocity'].to(device, dtype=torch.float)

        gt_waypoints = [
            torch.stack(batch['waypoints'][j], dim=1).to(device, dtype=torch.float)
            for j in range(config.pred_len)
        ]
        gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float)

        # --------------------------
        # 2. Forward pass
        # --------------------------
        pred_segs, pred_wp, _ = self(rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, gt_velocity)

        

        # --------------------------
        # 3. Compute losses
        # --------------------------
        loss_seg = 0
        for i in range(config.seq_len):
            loss_seg += utility.BCEDice(pred_segs[i], segs[i])
        loss_seg /= config.seq_len

        loss_wp = F.l1_loss(pred_wp, gt_waypoints)

        # Get current loss weights
        params_lw = opt_lw.param_groups[0]['params']
        total_loss = params_lw[0] * loss_seg + params_lw[1] * loss_wp

        # --------------------------
        # 4. Backpropagation
        # --------------------------
        opt.zero_grad(set_to_none=True)

        if batch_idx == 0:
            # Store initial losses for GradNorm
            self.loss_seg_0 = loss_seg.detach()
            self.loss_wp_0 = loss_wp.detach()
            self.manual_backward(total_loss)

        elif batch_idx < len(self.trainer.train_dataloader) - 1:
            self.manual_backward(total_loss)

        else:
            # Last batch of the epoch → GradNorm update
            if config.MGN:
                opt_lw.zero_grad(set_to_none=True)
                self.manual_backward(total_loss, retain_graph=True)

                params = list(filter(lambda p: p.requires_grad, self.parameters()))

                # Gradient norms for each task
                G0R = torch.autograd.grad(loss_seg, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
                G0 = torch.norm(G0R[0], keepdim=True)
                G1R = torch.autograd.grad(loss_wp, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
                G1 = torch.norm(G1R[0], keepdim=True)

                G_avg = (G0 + G1) / len(config.loss_weights)

                # Relative losses
                loss_seg_hat = loss_seg / (self.loss_seg_0 + 1e-8)
                loss_wp_hat = loss_wp / (self.loss_wp_0 + 1e-8)
                loss_hat_avg = (loss_seg_hat + loss_wp_hat) / len(config.loss_weights)

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

        opt.step()

        # --------------------------
        # 5. Log losses
        # --------------------------
        total_loss_meter.update(total_loss.item())
        seg_loss_meter.update(loss_seg.item())
        wp_loss_meter.update(loss_wp.item())

        self.log("train_total_loss", total_loss_meter.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_seg_loss", seg_loss_meter.avg, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_wp_loss", wp_loss_meter.avg, prog_bar=False, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        device = self.device
        config = self.config

        # --------------------------
        # 1. Prepare input tensors
        # --------------------------
        rgbs, segs, pt_cloud_xs, pt_cloud_zs = [], [], [], []

        for i in range(config.seq_len):
            rgbs.append(batch['rgbs'][i].to(device, dtype=torch.float))
            segs.append(batch['segs'][i].to(device, dtype=torch.float))
            pt_cloud_xs.append(batch['pt_cloud_xs'][i].to(device, dtype=torch.float))
            pt_cloud_zs.append(batch['pt_cloud_zs'][i].to(device, dtype=torch.float))

        rp1 = torch.stack(batch['rp1'], dim=1).to(device, dtype=torch.float)
        rp2 = torch.stack(batch['rp2'], dim=1).to(device, dtype=torch.float)
        gt_velocity = batch['velocity'].to(device, dtype=torch.float)

        gt_waypoints = [
            torch.stack(batch['waypoints'][j], dim=1).to(device, dtype=torch.float)
            for j in range(config.pred_len)
        ]
        gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float)

        # --------------------------
        # 2. Forward pass (no_grad)
        # --------------------------
        with torch.no_grad():
            pred_segs, pred_wp, _ = self(rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, gt_velocity)

            # --------------------------
            # 3. Compute losses
            # --------------------------
            loss_seg = 0
            for i in range(config.seq_len):
                loss_seg += utility.BCEDice(pred_segs[i], segs[i])
            loss_seg /= config.seq_len

            loss_wp = F.l1_loss(pred_wp, gt_waypoints)

            total_loss = loss_seg + loss_wp

        # --------------------------
        # 4. Log metrics
        # --------------------------
        self.log("val_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_seg_loss", loss_seg, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_wp_loss", loss_wp, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return {
            "val_total_loss": total_loss.detach(),
            "val_seg_loss": loss_seg.detach(),
            "val_wp_loss": loss_wp.detach(),
        }   

    def test_step(self, batch, batch_idx):
        device = self.device
        config = self.config

        # --------------------------
        # 1. Prepare input tensors
        # --------------------------
        rgbs, segs, pt_cloud_xs, pt_cloud_zs = [], [], [], []

        for i in range(config.seq_len):
            rgbs.append(batch['rgbs'][i].to(device, dtype=torch.float))
            segs.append(batch['segs'][i].to(device, dtype=torch.float))
            pt_cloud_xs.append(batch['pt_cloud_xs'][i].to(device, dtype=torch.float))
            pt_cloud_zs.append(batch['pt_cloud_zs'][i].to(device, dtype=torch.float))

        rp1 = torch.stack(batch['rp1'], dim=1).to(device, dtype=torch.float)
        rp2 = torch.stack(batch['rp2'], dim=1).to(device, dtype=torch.float)
        gt_velocity = torch.stack(batch['velocity'], dim=1).to(device, dtype=torch.float)

        gt_waypoints = [
            torch.stack(batch['waypoints'][j], dim=1).to(device, dtype=torch.float)
            for j in range(config.pred_len)
        ]
        gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float)

        # --------------------------
        # 2. Forward pass (no_grad)
        # --------------------------
        with torch.no_grad():
            pred_segs, pred_wp, _ = self(rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, gt_velocity)

            # --------------------------
            # 3. Compute losses
            # --------------------------
            loss_seg = 0
            for i in range(config.seq_len):
                loss_seg += utility.BCEDice(pred_segs[i], segs[i])
            loss_seg /= config.seq_len

            loss_wp = F.l1_loss(pred_wp, gt_waypoints)
            total_loss = loss_seg + loss_wp

        # --------------------------
        # 4. Log metrics (for test set)
        # --------------------------
        self.log("test_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_seg_loss", loss_seg, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_wp_loss", loss_wp, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return {
            "test_total_loss": total_loss.detach(),
            "test_seg_loss": loss_seg.detach(),
            "test_wp_loss": loss_wp.detach(),
        }

    def test_epoch_end(self, outputs):
        avg_total = torch.stack([x["test_total_loss"] for x in outputs]).mean()
        avg_seg = torch.stack([x["test_seg_loss"] for x in outputs]).mean()
        avg_wp = torch.stack([x["test_wp_loss"] for x in outputs]).mean()

        self.log("test_total_loss_epoch", avg_total, prog_bar=True)
        self.log("test_seg_loss_epoch", avg_seg)
        self.log("test_wp_loss_epoch", avg_wp)


    def configure_optimizers(self):
        optima = optim.AdamW(self.parameters(), weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.1, patience=4, min_lr=1e-6)

        #optimizer lw
        params_lw = [
            torch.tensor([self.config.loss_weights[i]], 
                 dtype=torch.float32, 
                 device='cuda').requires_grad_(True)
            for i in range(len(self.config.loss_weights))]
        optima_lw = optim.SGD(params_lw, lr=self.config.lr)

        return [
            {
                "optimizer": optima,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_total_loss"},
            },
            {
                "optimizer": optima_lw,
            },
        ]


