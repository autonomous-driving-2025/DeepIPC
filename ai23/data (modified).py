import os
import yaml
import cv2
from collections import deque
import numpy as np
import torch 
from torch.utils.data import Dataset


# ============== HELPER FUNCTIONS FOR POINT CLOUD ==============

def structured_to_unstructured(arr):
    """
    Convert a structured numpy array (with named fields x, y, z) to regular (N, 3) array.
    """
    if arr.dtype.names is not None:
        # It's a structured array with named fields
        if 'x' in arr.dtype.names and 'y' in arr.dtype.names and 'z' in arr.dtype.names:
            return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
        else:
            # Use first 3 fields
            fields = list(arr.dtype.names)[:3]
            return np.column_stack([arr[f] for f in fields]).astype(np.float32)
    return arr.astype(np.float32)


def project_points_to_image(points, height=720, width=1280, fx=700, fy=700, cx=640, cy=360):
    """
    Project unorganized point cloud (N, 3) to organized image format (H, W, 3).
    Uses pinhole camera model with approximate ZED 2i intrinsics.
    
    Args:
        points: (N, 3) array of x, y, z points in camera frame
        height, width: output image dimensions
        fx, fy: focal lengths
        cx, cy: principal point
    
    Returns:
        (H, W, 3) array with x, y, z values at each pixel
    """
    pt_image = np.zeros((height, width, 3), dtype=np.float32)
    
    if points.size == 0 or len(points) == 0:
        return pt_image
    
    # Ensure points is (N, 3)
    if len(points.shape) == 1:
        if points.size % 3 == 0:
            points = points.reshape(-1, 3)
        else:
            return pt_image
    
    points = points.astype(np.float32)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Filter valid depth (positive Z values, reasonable range)
    valid = (z > 0.1) & (z < 100) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    
    if len(z) == 0:
        return pt_image
    
    # Project to image coordinates (pinhole camera model)
    # u = fx * X/Z + cx, v = fy * Y/Z + cy
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)
    
    # Filter within image bounds
    valid_proj = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[valid_proj], v[valid_proj]
    x_proj, y_proj, z_proj = x[valid_proj], y[valid_proj], z[valid_proj]
    
    if len(z_proj) == 0:
        return pt_image
    
    # Fill image - sort by depth (far to near) so nearer points overwrite
    sort_idx = np.argsort(-z_proj)
    u, v = u[sort_idx], v[sort_idx]
    x_proj, y_proj, z_proj = x_proj[sort_idx], y_proj[sort_idx], z_proj[sort_idx]
    
    pt_image[v, u, 0] = x_proj
    pt_image[v, u, 1] = y_proj
    pt_image[v, u, 2] = z_proj
    
    return pt_image


# ============== MAIN DATASET CLASS ==============

class WHILL_Data(Dataset):

    def __init__(self, data_root, conditions, config):
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.data_rate = config.data_rate
        self.rp1_close = config.rp1_close

        self.condition = []
        self.route = []
        self.filename = []
        self.rgb = []
        self.seg = []
        self.pt_cloud = []
        self.lon = []
        self.lat = []
        self.loc_x = []
        self.loc_y = []
        self.rp1_lon = []
        self.rp1_lat = []
        self.rp2_lon = []
        self.rp2_lat = []
        self.bearing = []
        self.loc_heading = []
        self.steering = []
        self.throttle = []
        self.velocity_l = []
        self.velocity_r = []
        
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
                root_files.sort()
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    
                    # load route list
                    with open(route_dir+"/"+route+"_routepoint_list.yml", 'r') as rp_listx:
                        rp_list = yaml.load(rp_listx, Loader=yaml.FullLoader)
                        rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
                        rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])
                    
                    # list and sort files
                    files = os.listdir(route_dir+"/camera/rgb/")
                    files.sort()

                    # MAF buffer
                    sin_angle_buff = deque()
                    if self.config.n_buffer != 0:
                        for _ in range(0, self.config.n_buffer*self.data_rate-1):
                            sin_angle_buff.append(0.0)

                    for i in range(0, len(files)-(self.seq_len-1)-(self.pred_len*self.data_rate)):
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
                            pt_clouds.append(route_dir+"/camera/depth/cld2/"+filename[:-3]+"npy")

                        preload_rgb.append(rgbs)
                        preload_seg.append(segs)
                        preload_pt_cloud.append(pt_clouds)

                        # metadata
                        preload_condition.append(condition)
                        preload_route.append(route)
                        preload_filename.append(filename)

                        # get current frame metadata
                        with open(route_dir+"/meta/"+filename[:-3]+"yml", "r") as read_meta_current:
                            meta_current = yaml.load(read_meta_current, Loader=yaml.FullLoader)
                        
                        loc_xs.append(meta_current['local_position_xyz'][0])
                        loc_ys.append(meta_current['local_position_xyz'][1])
                        
                        # Calculate heading from quaternion
                        qx, qy, qz, qw = meta_current['local_orientation_xyzw']
                        yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
                        loc_headings.append(yaw)
                        
                        # GPS coordinates
                        preload_lon.append(meta_current['global_position_latlon'][1])
                        preload_lat.append(meta_current['global_position_latlon'][0])

                        # Bearing from global orientation
                        gqx, gqy, gqz, gqw = meta_current['global_orientation_xyzw']
                        bearing_rad = np.arctan2(2.0*(gqw*gqz + gqx*gqy), 1.0 - 2.0*(gqy*gqy + gqz*gqz))
                        bearing_deg = np.degrees(bearing_rad)
                        
                        sin_angle_buff.append(np.sin(bearing_rad))
                        sin_angle_buff_mean = np.array(sin_angle_buff).mean()
                        
                        # Apply quadrant correction
                        if 0 < bearing_deg <= 90:
                            bearing_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif 90 < bearing_deg <= 180:
                            bearing_deg_maf = 180 - np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif -180 < bearing_deg <= -90:
                            bearing_deg_maf = -180 - np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif -90 < bearing_deg <= 0:
                            bearing_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
                        else:
                            bearing_deg_maf = bearing_deg
                            
                        sin_angle_buff.popleft()
                        bearing_robot_deg = bearing_deg_maf + self.config.bearing_bias

                        if bearing_robot_deg > 180:
                            bearing_robot_deg = bearing_robot_deg - 360
                        elif bearing_robot_deg < -180:
                            bearing_robot_deg = bearing_robot_deg + 360
                        preload_bearing.append(np.radians(bearing_robot_deg))

                        # Vehicular controls
                        preload_steering.append(0.0)
                        preload_throttle.append(meta_current['velocity'])
                        preload_velocity_l.append(meta_current['velocity'])
                        preload_velocity_r.append(meta_current['velocity'])

                        # Assign next route lat lon
                        about_to_finish = False
                        for r in range(2):
                            next_lat = rp_list['route_point']['latitude'][r]
                            next_lon = rp_list['route_point']['longitude'][r]
                            dLat_m = (next_lat-meta_current['global_position_latlon'][0]) * 40008000 / 360
                            dLon_m = (next_lon-meta_current['global_position_latlon'][1]) * 40075000 * np.cos(np.radians(meta_current['global_position_latlon'][0])) / 360
                            
                            if r==0 and np.sqrt(dLat_m**2 + dLon_m**2) <= self.rp1_close and not about_to_finish:
                                if len(rp_list['route_point']['latitude']) > 2:
                                    rp_list['route_point']['latitude'].pop(0)
                                    rp_list['route_point']['longitude'].pop(0)
                                else:
                                    about_to_finish = True
                                    rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                                    rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]

                                next_lat = rp_list['route_point']['latitude'][r]
                                next_lon = rp_list['route_point']['longitude'][r]
                            
                            if r==0:
                                preload_rp1_lon.append(next_lon)
                                preload_rp1_lat.append(next_lat)
                            else:
                                preload_rp2_lon.append(next_lon)
                                preload_rp2_lat.append(next_lat)

                        # Read future frames
                        for k in range(1, self.pred_len+1):
                            filenamef = files[(i+self.seq_len-1) + (k*self.data_rate)]
                            with open(route_dir+"/meta/"+filenamef[:-3]+"yml", "r") as read_meta_future:
                                meta_future = yaml.load(read_meta_future, Loader=yaml.FullLoader)
                            loc_xs.append(meta_future['local_position_xyz'][0])
                            loc_ys.append(meta_future['local_position_xyz'][1])
                            
                            fqx, fqy, fqz, fqw = meta_future['local_orientation_xyzw']
                            future_yaw = np.arctan2(2.0*(fqw*fqz + fqx*fqy), 1.0 - 2.0*(fqy*fqy + fqz*fqz))
                            loc_headings.append(future_yaw)

                        preload_loc_x.append(loc_xs)
                        preload_loc_y.append(loc_ys)
                        preload_loc_heading.append(loc_headings)

                # Save to npy
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

            # Load from npy
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
            self.steering += preload_dict.item()['steering']
            self.throttle += preload_dict.item()['throttle']
            self.velocity_l += preload_dict.item()['velocity_l']
            self.velocity_r += preload_dict.item()['velocity_r']
            print("Preloading " + str(len(preload_dict.item()['rgb'])) + " sequences from " + preload_file)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        data = dict()
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
            # RGB images
            rgb_img = cv2.imread(seq_rgbs[i])
            rgb_cropped = crop_matrix(rgb_img, resize=self.config.scale, crop=self.config.crop_roi)
            data['rgbs'].append(torch.from_numpy(rgb_cropped.transpose(2, 0, 1).astype(np.float32)))
            
            # Segmentation
            seg_img = cv2.imread(seq_segs[i])
            seg_cropped = crop_matrix(seg_img, resize=self.config.scale, crop=self.config.crop_roi)
            seg_onehot = cls2one_hot(seg_cropped, n_class=self.config.n_class)
            data['segs'].append(torch.from_numpy(seg_onehot.astype(np.float32)))

            # Camera point cloud - load structured array and project to image
            pc_raw = np.load(seq_pt_clouds[i], allow_pickle=True)
            
            # Convert structured array to regular (N, 3) array
            if pc_raw.dtype.names is not None:
                pc_unstructured = structured_to_unstructured(pc_raw)
            else:
                pc_unstructured = pc_raw
            
            # Project to organized image format (H, W, 3)
            pc_image = project_points_to_image(pc_unstructured, height=720, width=1280)
            
            # Crop and process
            pc_cropped = crop_matrix(pc_image, resize=self.config.scale, crop=self.config.crop_roi)
            pt_cloud = np.nan_to_num(pc_cropped.transpose(2, 0, 1), nan=0.0, posinf=39.99999, neginf=0.2)
            
            data['pt_cloud_xs'].append(torch.from_numpy(pt_cloud[0:1, :, :].astype(np.float32)))
            data['pt_cloud_zs'].append(torch.from_numpy(pt_cloud[2:3, :, :].astype(np.float32)))

        # Current ego robot position and heading at index 0
        ego_loc_x = seq_loc_xs[0]
        ego_loc_y = seq_loc_ys[0]
        ego_loc_heading = seq_loc_headings[0]   

        # Waypoint processing to local coordinates
        data['waypoints'] = []
        for j in range(1, self.pred_len+1):
            local_waypoint = transform_2d_points(
                np.zeros((1, 3)), 
                np.pi/2 - seq_loc_headings[j], 
                seq_loc_xs[j], 
                seq_loc_ys[j], 
                np.pi/2 - ego_loc_heading, 
                ego_loc_x, 
                ego_loc_y
            )
            data['waypoints'].append(tuple(local_waypoint[0, :2]))

        # Convert rp1_lon, rp1_lat, rp2_lon, rp2_lat to local coordinates
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([
            [np.cos(bearing_robot), -np.sin(bearing_robot)],
            [np.sin(bearing_robot),  np.cos(bearing_robot)]
        ])
        dLat1_m = (self.rp1_lat[index] - lat_robot) * 40008000 / 360
        dLon1_m = (self.rp1_lon[index] - lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360
        dLat2_m = (self.rp2_lat[index] - lat_robot) * 40008000 / 360
        dLon2_m = (self.rp2_lon[index] - lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360
        data['rp1'] = tuple(R_matrix.T.dot(np.array([dLon1_m, dLat1_m])))
        data['rp2'] = tuple(R_matrix.T.dot(np.array([dLon2_m, dLat2_m])))

        # Vehicular controls
        data['steering'] = self.steering[index]
        data['throttle'] = self.throttle[index]
        data['lr_velo'] = tuple(np.array([self.velocity_l[index], self.velocity_r[index]]))

        # Metadata for testing
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data


# ============== UTILITY FUNCTIONS ==============

def swap_RGB2BGR(matrix):
    red = matrix[:, :, 0].copy()
    blue = matrix[:, :, 2].copy()
    matrix[:, :, 0] = blue
    matrix[:, :, 2] = red
    return matrix


def crop_matrix(image, resize=1, D3=True, crop=[512, 1024]):
    upper_left_yx = [int((image.shape[0]/2) - (crop[0]/2)), int((image.shape[1]/2) - (crop[1]/2))]
    if D3:
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1], :]
    else:
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1]]

    WH_resized = (int(cropped_im.shape[1]/resize), int(cropped_im.shape[0]/resize))
    resized_image = cv2.resize(cropped_im, WH_resized, interpolation=cv2.INTER_NEAREST)

    return resized_image


def cls2one_hot(ss_gt, n_class):
    ss_gt = np.transpose(ss_gt, (2, 0, 1))
    ss_gt = ss_gt[:1, :, :].reshape(ss_gt.shape[1], ss_gt.shape[2])
    result = (np.arange(n_class) == ss_gt[..., None]).astype(int)
    result = np.transpose(result, (2, 0, 1))
    return result


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    out[:, 2] = xyz[:, 2]

    return out