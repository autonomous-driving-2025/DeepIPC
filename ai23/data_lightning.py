import os
import yaml
import cv2
from PIL import Image, ImageFile
from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import utility

class KARR_Dataset(Dataset):
    def __init__(self, data_root, conditions, config):
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.data_rate = config.data_rate
        self.rp1_close = config.rp1_close

        self.condition = [] #buat offline test nantinya
        self.route = []
        self.filename = []
        self.rgb = []
        self.seg = []
        self.pt_cloud = []
        self.lon = []
        self.lat = []
        self.loc_x = []
        self.loc_y = []
        self.loc_heading = []
        self.rp1_lon = []
        self.rp1_lat = []
        self.rp2_lon = []
        self.rp2_lat = []
        self.bearing = []
        self.velocity = []

        for condition in conditions:
            sub_root = os.path.join(data_root, condition)
            preload_file = os.path.join(sub_root, 'xr14_seq'+str(self.seq_len)+'_pred'+str(self.pred_len)+'_rp1'+str(self.rp1_close)+'_maf'+str(self.config.n_buffer*self.data_rate)+'.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_condition = []
                preload_route = []
                preload_filename = []
                preload_rgb = []
                preload_seg = []
                preload_pt_cloud = []
                preload_lon = []
                preload_lat = []
                preload_loc_x = []
                preload_loc_y = []
                preload_rp1_lon = []
                preload_rp1_lat = []
                preload_rp2_lon = []
                preload_rp2_lat = []
                preload_bearing = []
                preload_loc_heading = []
                preload_steering = []
                preload_throttle = []
                preload_velocity_l = []
                preload_velocity_r = []

                # list sub-directories in root
                root_files = os.listdir(sub_root)
                root_files.sort() #nanti sudah diacak oleh torch dataloader
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)

                    #load route list nya
                    with open(route_dir+"/"+route+"_routepoint_list.yml", 'r') as rp_listx:
                    # with open(route_dir+"/gmaps"+route[-2:]+"_routepoint_list.yml", 'r') as rp_listx:
                        rp_list = yaml.load(rp_listx, Loader=yaml.FullLoader)
                        #assign end point sebagai route terakhir
                        rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
                        rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])

                    #list dan sort file, slah satu saja
                    files = os.listdir(route_dir+"/camera/rgb/")
                    files.sort() #nanti sudah diacak oleh torch dataloader

                    # buat MAF mean avg filter
                    sin_angle_buff = deque()
                    if self.config.n_buffer != 0:
                        for _ in range(0, self.config.n_buffer*self.data_rate-1):
                            sin_angle_buff.append(0.0)

                    for i in range(0, len(files)-(self.seq_len-1)-(self.pred_len*self.data_rate)): #kurangi sesuai dengan jumlah sequence dan wp yang akan diprediksi
                        #ini yang buat yg disequence kan
                        rgbs = []
                        segs = []
                        pt_clouds = []
                        loc_xs = []
                        loc_ys = []
                        loc_headings = []

                        # read files sequentially (past and current frames)
                        for j in range(0, self.seq_len):
                            filename = files[i+j]
                            rgbs.append(route_dir+"/camera/rgb/"+filename)
                            segs.append(route_dir+"/camera/seg/img/"+filename)
                            # pt_clouds.append(route_dir+"/point_cloud/"+filename[:-3]+"npy")
                            pt_clouds.append(route_dir+"/camera/depth/cld2/"+filename[:-3]+"npy")

                        #appendkan
                        preload_rgb.append(rgbs)
                        preload_seg.append(segs)
                        preload_pt_cloud.append(pt_clouds)

                        #metadata buat testing nantinya
                        preload_condition.append(condition)
                        preload_route.append(route)
                        preload_filename.append(filename)

                        # ambil local loc, heading, vehicular controls, gps loc, dan bearing pada seq terakhir saja (current)
                        with open(route_dir+"/meta/"+filename[:-3]+"yml", "r") as read_meta_current:
                            meta_current = yaml.load(read_meta_current, Loader=yaml.FullLoader)
                        loc_xs.append(meta_current['local_position_xyz'][0])
                        loc_ys.append(meta_current['local_position_xyz'][1])
                        # Calculate heading from quaternion
                        qx, qy, qz, qw = meta_current['local_orientation_xyzw']
                        yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
                        loc_headings.append(yaw)

                        preload_lon.append(meta_current['global_position_latlon'][1])
                        preload_lat.append(meta_current['global_position_latlon'][0])


                        #apply MAF ke bearing
                        gqx, gqy, gqz, gqw = meta_current['global_orientation_xyzw']
                        bearing_rad = np.arctan2(2.0*(gqw*gqz + gqx*gqy), 1.0 - 2.0*(gqy*gqy + gqz*gqz))
                        bearing_deg = np.degrees(bearing_rad)

                        sin_angle_buff.append(np.sin(bearing_rad))
                        sin_angle_buff_mean = np.array(sin_angle_buff).mean()

                        #cek kuadran
                        if 0 < bearing_deg <= 90: #Q1
                            bearing_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif 90 < bearing_deg <= 180: #Q2
                            bearing_deg_maf = 180 - np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif -180 < bearing_deg <= -90: #Q3 180 - 270
                            bearing_deg_maf = -180 - np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif -90 < bearing_deg <= 0: #Q4 270 - 360
                            bearing_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
                        sin_angle_buff.popleft() #hilangkan 1 untuk diisilagi dengan next data nantinya
                        bearing_robot_deg = bearing_deg_maf+self.config.bearing_bias

                        if bearing_robot_deg > 180: #buat jadi -180 ke 0
                            bearing_robot_deg = bearing_robot_deg - 360
                        elif bearing_robot_deg < -180: #buat jadi 180 ke 0
                            bearing_robot_deg = bearing_robot_deg + 360
                        preload_bearing.append(np.radians(bearing_robot_deg))

                        #vehicular controls
                        preload_steering.append(meta_current['steering'])
                        preload_throttle.append(meta_current['throttle'])
                        preload_velocity_l.append(meta_current['velocity'])
                        preload_velocity_r.append(meta_current['velocity'])

                        #assign next route lat lon
                        about_to_finish = False
                        for r in range(2): #ada 2 route point
                            next_lat = rp_list['route_point']['latitude'][r]
                            next_lon = rp_list['route_point']['longitude'][r]
                            dLat_m = (next_lat-meta_current['global_position_latlon'][0]) * 40008000 / 360 #111320 #Y
                            dLon_m = (next_lon-meta_current['global_position_latlon'][1]) * 40075000 * np.cos(np.radians(meta_current['global_position_latlon'])) / 360 #X

                            if r==0 and np.any(np.sqrt(dLat_m**2 + dLon_m**2) <= self.rp1_close) and not about_to_finish: #jika jarak euclidian rp1 <= jarak min, hapus route dan loncat ke next route
                                if len(rp_list['route_point']['latitude']) > 2: #jika jumlah route list masih > 2
                                    rp_list['route_point']['latitude'].pop(0)
                                    rp_list['route_point']['longitude'].pop(0)
                                else: #berarti mendekati finish
                                    about_to_finish = True
                                    rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                                    rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]

                                next_lat = rp_list['route_point']['latitude'][r]
                                next_lon = rp_list['route_point']['longitude'][r]

                            if r==0:
                                preload_rp1_lon.append(next_lon)
                                preload_rp1_lat.append(next_lat)
                            else: #r==1
                                preload_rp2_lon.append(next_lon)
                                preload_rp2_lat.append(next_lat)


                        # read files sequentially (future frames)
                        for k in range(1, self.pred_len+1):
                            filenamef = files[(i+self.seq_len-1) + (k*self.data_rate)] #future seconds, makanya dikali data rate
                            # meta
                            with open(route_dir+"/meta/"+filenamef[:-3]+"yml", "r") as read_meta_future:
                                meta_future = yaml.load(read_meta_future, Loader=yaml.FullLoader)
                            loc_xs.append(meta_future['local_position_xyz'][0])
                            loc_ys.append(meta_future['local_position_xyz'][1])

                            fqx, fqy, fqz, fqw = meta_future['local_orientation_xyzw']
                            future_yaw = np.arctan2(2.0*(fqw*fqz + fqx*fqy), 1.0 - 2.0*(fqy*fqy + fqz*fqz))
                            loc_headings.append(future_yaw)

                        #append sisanya
                        preload_loc_x.append(loc_xs)
                        preload_loc_y.append(loc_ys)
                        preload_loc_heading.append(loc_headings)


                # dump ke npy
                preload_dict = {}
                preload_dict['condition'] = preload_condition
                preload_dict['route'] = preload_route
                preload_dict['filename'] = preload_filename
                preload_dict['rgb'] = preload_rgb
                preload_dict['seg'] = preload_seg
                preload_dict['pt_cloud'] = preload_pt_cloud
                preload_dict['lon'] = preload_lon
                preload_dict['lat'] = preload_lat
                preload_dict['loc_x'] = preload_loc_x
                preload_dict['loc_y'] = preload_loc_y
                preload_dict['rp1_lon'] = preload_rp1_lon
                preload_dict['rp1_lat'] = preload_rp1_lat
                preload_dict['rp2_lon'] = preload_rp2_lon
                preload_dict['rp2_lat'] = preload_rp2_lat
                preload_dict['bearing'] = preload_bearing
                preload_dict['loc_heading'] = preload_loc_heading
                preload_dict['steering'] = preload_steering
                preload_dict['throttle'] = preload_throttle
                preload_dict['velocity_l'] = preload_velocity_l
                preload_dict['velocity_r'] = preload_velocity_r
                np.save(preload_file, preload_dict)


            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)

            self.condition += preload_dict.item()['condition']
            self.route += preload_dict.item()['route']
            self.filename += preload_dict.item()['filename']
            self.rgb += preload_dict.item()['rgb']
            self.seg += preload_dict.item()['seg']
            self.pt_cloud += preload_dict.item()['pt_cloud']
            self.lon += preload_dict.item()['lon']
            self.lat += preload_dict.item()['lat']
            self.loc_x += preload_dict.item()['loc_x']
            self.loc_y += preload_dict.item()['loc_y']
            self.rp1_lon += preload_dict.item()['rp1_lon']
            self.rp1_lat += preload_dict.item()['rp1_lat']
            self.rp2_lon += preload_dict.item()['rp2_lon']
            self.rp2_lat += preload_dict.item()['rp2_lat']
            self.bearing += preload_dict.item()['bearing']
            self.loc_heading += preload_dict.item()['loc_heading']
            print("TYPE:", type(preload_dict))
            print("ITEM TYPE:", type(preload_dict.item()))
            print("KEYS:", preload_dict.item().keys())
            print("Loading from:", preload_file)


            self.steering += preload_dict.item()['steering']
            self.throttle += preload_dict.item()['throttle']
            self.velocity_l += preload_dict.item()['velocity_l']
            self.velocity_r += preload_dict.item()['velocity_r']
            print("Preloading " + str(len(preload_dict.item()['rgb'])) + " sequences from " + preload_file)


    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        data = dict()
        #metadata buat testing nantinya
        data['condition'] = self.condition[index]
        data['route'] = self.route[index]
        data['filename'] = self.filename[index]

        data['rgbs'] = []
        data['segs'] = []
        data['pt_cloud_xs'] = []
        data['pt_cloud_zs'] = []

        seq_rgbs = self.rgb[index]
        seq_segs = self.seg[index]
        seq_pt_clouds = self.pt_cloud[index]
        seq_loc_xs = self.loc_x[index]
        seq_loc_ys = self.loc_y[index]
        seq_loc_headings = self.loc_heading[index]

        for i in range(0, self.seq_len):
        #RGB
            data['rgbs'].append(torch.from_numpy(np.array(crop_matrix(cv2.imread(seq_rgbs[i]), resize=self.config.scale, crop=self.config.crop_roi).transpose(2,0,1))))
        #SEG
            data['segs'].append(torch.from_numpy(np.array(cls2one_hot(crop_matrix(cv2.imread(seq_segs[i]), resize=self.config.scale, crop=self.config.crop_roi), n_class=self.config.n_class))))

            pt_cloud = np.nan_to_num(crop_matrix(np.load(seq_pt_clouds[i])[:,:,0:3], resize=self.config.scale, crop=self.config.crop_roi).transpose(2,0,1), nan=0.0, posinf=39.99999, neginf=0.2) #min_d, max_d, -max_d, ambil xyz-nya saja 0:3, baca https://www.stereolabs.com/docs/depth-sensing/depth-settings/
            # data['pt_cloud_xs'].append(torch.from_numpy(np.array(pt_cloud[0:1,:,:])))
            # data['pt_cloud_zs'].append(torch.from_numpy(np.array(pt_cloud[2:3,:,:])))
            data['pt_cloud_xs'].append(torch.from_numpy(pt_cloud[0:1, :, :].astype(np.float32)))
            data['pt_cloud_zs'].append(torch.from_numpy(pt_cloud[2:3, :, :].astype(np.float32)))


        #current ego robot position dan heading di index 0
        ego_loc_x = seq_loc_xs[0]
        ego_loc_y = seq_loc_ys[0]
        ego_loc_heading = seq_loc_headings[0]

        # waypoint processing to local coordinates
        data['waypoints'] = [] #wp dalam local coordinate
        for j in range(1, self.pred_len+1):
            local_waypoint = transform_2d_points(np.zeros((1,3)),
                np.pi/2-seq_loc_headings[j], seq_loc_xs[j], seq_loc_ys[j], np.pi/2-ego_loc_heading, ego_loc_x, ego_loc_y)
            data['waypoints'].append(tuple(local_waypoint[0,:2]))


        # convert rp1_lon, rp1_lat rp2_lon, rp2_lat ke local coordinates
        #komputasi dari global ke local
        #https://gamedev.stackexchange.com/questions/79765/how-do-i-convert-from-the-global-coordinate-space-to-a-local-space
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([[np.cos(bearing_robot), -np.sin(bearing_robot)],
                            [np.sin(bearing_robot),  np.cos(bearing_robot)]])
        dLat1_m = (self.rp1_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon1_m = (self.rp1_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        dLat2_m = (self.rp2_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon2_m = (self.rp2_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        data['rp1'] = tuple(R_matrix.T.dot(np.array([dLon1_m, dLat1_m])))
        data['rp2'] = tuple(R_matrix.T.dot(np.array([dLon2_m, dLat2_m])))

        # print("rp1_lat "+str(self.rp1_lat[index]))
        # print("rp2_lat "+str(self.rp2_lat[index]))
        # print("rp1_lon "+str(self.rp1_lon[index]))
        # print("rp2_lon "+str(self.rp2_lon[index]))

        #vehicular controls dan velocity jadikan satu LR
        data['steering'] = self.steering[index]
        data['throttle'] = self.throttle[index]
        data['lr_velo'] = tuple(np.array([self.velocity_l[index], self.velocity_r[index]]))

        #metadata buat testing nantinya
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data


class KARR_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, conditions, config):
        super().__init__()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

