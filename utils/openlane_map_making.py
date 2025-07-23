import copy
import json
import os

import cv2
import numpy as np
import torch
import pickle
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from utils.coord_util import ego2image,IPM2ego_matrix, ego2image_filtered
from utils.standard_camera_cpu import Standard_camera
from functools import cmp_to_key

class OpenLane_dataset_with_offset(Dataset):
    def __init__(self, image_paths, 
                   gt_paths,
                   map_paths,
                   x_range,
                   y_range, 
                   meter_per_pixel, 
                   data_trans,
                   output_2d_shape,
                  virtual_camera_config):
        self.heatmap_h = 18
        self.heatmap_w = 32
        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.map_paths = map_paths
        self.cnt_list = []
        self.lane3d_thick = 1
        self.lane2d_thick = 3
        self.lane_length_threshold = 3  #
        card_list = os.listdir(self.gt_paths)
        for card in card_list:
            gt_paths = os.path.join(self.gt_paths, card)
            gt_list = os.listdir(gt_paths)
            for cnt in gt_list:
                self.cnt_list.append([card, cnt])

        ''' virtual camera paramter'''
        self.use_virtual_camera = virtual_camera_config['use_virtual_camera']
        self.vc_intrinsic = virtual_camera_config['vc_intrinsic']
        self.vc_extrinsics = virtual_camera_config['vc_extrinsics']
        self.vc_image_shape = virtual_camera_config['vc_image_shape']

        ''' transform loader '''
        self.output2d_size = output_2d_shape
        self.trans_image = data_trans
        self.ipm_h, self.ipm_w = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
            (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)
        
    def project_bev_height_map_to_image_plane(self, bev_height_map, extrinsic_matrix, intrinsic_matrix):
        # Create homogeneous coordinates for height map points
        points_3d = np.ones((self.ipm_h *  self.ipm_w, 4))
        points_3d[:, :2] = np.mgrid[self.x_range[0]:self.x_range[0] + self.ipm_h, self.y_range[0]*2:self.y_range[1]*2].T.reshape(-1, 2)*self.meter_per_pixel
        # Reshape height map to 1D array
        points_3d[:,2] = bev_height_map.flatten()

        # Project 3D points to image plane
        image_points = intrinsic_matrix @ extrinsic_matrix[:3,:] @ points_3d.T

        # Normalize image points
        image_points_normalized = image_points[:2] / image_points[2]
        # print(image_points_normalized)
        return image_points_normalized
    
    def project_point_cloud_to_image_plane(self, point_cloud, extrinsic, intrinsic):
        # Homogeneous coordinates for 3D point cloud
        point_cloud_homo = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

        # Project 3D points to image plane
        image_points_homogeneous = intrinsic @ extrinsic[:3,:] @ point_cloud_homo.T

        # Convert homogeneous coordinates to Cartesian coordinates
        image_points_cartesian = image_points_homogeneous[:2] / image_points_homogeneous[2]

        return image_points_cartesian
    
    def extract_heightmap_within_range(self, pc, vehicle_pose_matrix, cam_w_extrinsics, forward_range, lateral_range):
        gt_pc = np.concatenate(pc, axis=1)

        # Extract rotation matrix and translation vector from vehicle pose matrix
        R = vehicle_pose_matrix[:3, :3]  # Rotation matrix
        T = vehicle_pose_matrix[:3, 3]   # Translation vector

        R_inverse = np.linalg.inv(R)
        # Translate point cloud to vehicle position
        translated_point_cloud = gt_pc.T[:,:3] - T

        # Rotate point cloud to vehicle coordinate system
        point_cloud_vehicle = (R_inverse@translated_point_cloud.T).T
        
        # Extract rotation matrix and translation vector from vehicle pose matrix
        R_ex = cam_w_extrinsics[:3, :3]  # Rotation matrix
        T_ex = cam_w_extrinsics[:3, 3]   # Translation vector
        
        R_inverse_ex = np.linalg.inv(R_ex)
        # Translate point cloud to vehicle position
        translated_point_cloud_ex =point_cloud_vehicle[:,:3] - T_ex
        
        # Rotate point cloud to vehicle coordinate system
        point_cloud_camera = (R_inverse_ex@translated_point_cloud_ex.T).T

        x = point_cloud_camera[:, 0]
        y = point_cloud_camera[:, 1]

        # Filter points within the specified range
        mask = (x >= forward_range[0]) & (x <= forward_range[1]) & (y >=lateral_range[0]) & (y <= lateral_range[1])
        filtered_point_cloud = point_cloud_camera[mask]
        return filtered_point_cloud
        
    def generate_bev_height_map(self, point_cloud, resolution=(200,48), meter_per_pixel=0.5, vehicle_height=0):
        bev_resolution_x, bev_resolution_y = resolution
            
        # Initialize BEV height map
        bev_height_map = np.zeros((bev_resolution_x, bev_resolution_y))
        # 각 픽셀에 대한 Z 값의 합과 개수를 저장할 배열 초기화
        z_sum_array = np.zeros((bev_resolution_x, bev_resolution_y))
        count_array = np.zeros((bev_resolution_x, bev_resolution_y))
        
        # 포인트 클라우드 반복하면서 각 픽셀에 대한 Z 값의 합과 개수 계산
        x_coords = point_cloud[:, 0].astype(int)
        y_coords = point_cloud[:, 1].astype(int)
        # x_coords = resolution[0] - np.round((point_cloud[:, 0] - 3) / meter_per_pixel).astype(int) -1
        # y_coords = resolution[1] - np.round((point_cloud[:, 1] + 12) / meter_per_pixel).astype(int) -1
        z_values = point_cloud[:, 2]
        
        # 유효한 픽셀 좌표 찾기
        valid_pixels_mask = (x_coords >= 0) & (x_coords < bev_resolution_x) & (y_coords >= 0) & (y_coords < bev_resolution_y)

        # 각 픽셀에 대한 Z 값의 합과 개수 계산
        np.add.at(z_sum_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), z_values[valid_pixels_mask])
        np.add.at(count_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), 1)

        # 계산된 Z 값의 합을 개수로 나누어 평균 계산
        with np.errstate(divide='ignore', invalid='ignore'):  # 0으로 나누는 오류 무시
            bev_height_map = np.divide(z_sum_array, count_array, out=np.zeros_like(z_sum_array), where=count_array != 0)

        return bev_height_map
    
    def generate_binary_mask(self, bev_height_map):
        # Initialize binary mask
        binary_mask = np.zeros_like(bev_height_map, dtype=np.int)

        # Mark pixels with points as 1
        binary_mask[bev_height_map != 0] = 1

        return binary_mask
    
    def bev2ipm(self, bev, matrix_IPM2ego):
        ego_points = np.array([bev[0], bev[1]])
        ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (ego_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
        ipm_points_ = np.zeros_like(ipm_points)
        ipm_points_[0] = ipm_points[1]
        ipm_points_[1] = ipm_points[0]
        res_points = np.concatenate([ipm_points, np.array([bev[2]])], axis=0)
        return res_points        

    def get_seg_offset(self, idx, smooth=False):
        
        def gaussian2D(shape, sigma=1):
            m, n = [(ss - 1.) / 2. for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h
    
        
        gt_path = os.path.join(self.gt_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        # print(self.cnt_list[idx][0])
        map_path = os.path.join(self.map_paths, self.cnt_list[idx][0]+'.tfrecord_dbinfos.pkl')
        image_path = os.path.join(self.image_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'jpg'))
        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        with open(map_path,'rb') as map_f:
            map_db_infos = pickle.load(map_f)
            
        vehicle_pose = np.array(gt['pose'])
        map_segment_id = map_db_infos["segment_id"] 
        map_pointcloud = map_db_infos["global_pointcloud"]
        
        matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)  #

    
        cam_w_extrinsics = np.array(gt['extrinsic'])
        maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
                                            [-1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]], dtype=float)
        cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w  #
        R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        cam_extrinsics_persformer = copy.deepcopy(cam_w_extrinsics)
        cam_extrinsics_persformer[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), cam_extrinsics_persformer[:3, :3]),
            R_vg), R_gc)
        cam_extrinsics_persformer[0:2, 3] = 0.0
        matrix_lane2persformer = cam_extrinsics_persformer @ np.linalg.inv(maxtrix_camera2camera_w)

        cam_intrinsic = np.array(gt['intrinsic'])
        # print(cam_intrinsic)
        lanes = gt['lane_lines']
        
        roi_pointcloud = self.extract_heightmap_within_range(map_pointcloud, vehicle_pose, cam_w_extrinsics,self.x_range, self.y_range)
        del map_db_infos
        del map_pointcloud
        
        roi_pointcloud = np.vstack((roi_pointcloud.T, np.ones((1, roi_pointcloud.shape[0]))))
        new_roi = matrix_lane2persformer @ roi_pointcloud
        roi = np.array([new_roi[1], -new_roi[0], new_roi[2]])
        ipm_roi = self.bev2ipm(roi, matrix_IPM2ego)
        
        bev_heightmap = self.generate_bev_height_map(ipm_roi.T, resolution=(self.ipm_h, self.ipm_w), meter_per_pixel=self.meter_per_pixel)
        bev_heightmask = self.generate_binary_mask(bev_heightmap)
        bev_height_map_mask = np.stack((bev_heightmap, bev_heightmask), axis=0)

        # print("This image", image_w, image_h)
        image_gt = np.zeros((image_h, image_w), dtype=np.uint8)
        res_points_d = {}
        for idx in range(len(lanes)):
            lane1 = lanes[idx]
            lane_camera_w = np.array(lane1['xyz']).T[np.array(lane1['visibility']) >= 0.0].T
            lane_camera_w = np.vstack((lane_camera_w, np.ones((1, lane_camera_w.shape[1]))))
            lane_ego_persformer = matrix_lane2persformer @ lane_camera_w  #
            lane_ego_persformer[0], lane_ego_persformer[1] = lane_ego_persformer[1], -1 * lane_ego_persformer[0]
            lane_ego = cam_w_extrinsics @ lane_camera_w  #
            ''' plot uv '''
            uv1 = ego2image(lane_ego[:3], cam_intrinsic, cam_extrinsics)
            
            cv2.polylines(image_gt, [uv1[0:2, :].T.astype(int)], False, idx + 1, self.lane2d_thick)

            distance = np.sqrt((lane_ego_persformer[1][0] - lane_ego_persformer[1][-1]) ** 2 + (
                    lane_ego_persformer[0][0] - lane_ego_persformer[0][-1]) ** 2)
            if distance < self.lane_length_threshold:
                continue
            y = lane_ego_persformer[1]
            x = lane_ego_persformer[0]
            z = lane_ego_persformer[2]

            if smooth:
                if len(x) < 2:
                    continue
                elif len(x) == 2:
                    curve = np.polyfit(x, y, 1)
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) == 3:
                    curve = np.polyfit(x, y, 2)
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    curve = np.polyfit(x, y, 3)
                    function2 = interp1d(x, z, kind='cubic')
                x_base = np.linspace(min(x), max(x), 20)
                y_pred = np.poly1d(curve)(x_base)
                ego_points = np.array([x_base, y_pred])
                z = function2(x_base)
            else:
                ego_points = np.array([x, y])

            ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (ego_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)
            res_points_d[idx + 1] = res_points
        ipm_gt, offset_y_map, z_map = self.get_y_offset_and_z(res_points_d)
        # print(image.shape)
        ''' virtual camera '''
        if self.use_virtual_camera:
            sc = Standard_camera(self.vc_intrinsic, self.vc_extrinsics, self.vc_image_shape,
                                 cam_intrinsic, cam_extrinsics, (image.shape[0], image.shape[1]))
            trans_matrix = sc.get_matrix(height=0)
            image = cv2.warpPerspective(image, trans_matrix, self.vc_image_shape)
            image_gt = cv2.warpPerspective(image_gt, trans_matrix, self.vc_image_shape)
        return image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, bev_height_map_mask

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, bev_height_map_mask = self.get_seg_offset(idx)
        image_h, image_w, _ = image.shape
        cam_intrinsic[0] *= 800 / image_w
        cam_intrinsic[1] *= 600 / image_h
        transformed = self.trans_image(image=image)
        image = transformed["image"]
        intrinsic = torch.tensor(cam_intrinsic)
        ''' 2d gt'''
        image_gt = cv2.resize(image_gt, (self.output2d_size[1], self.output2d_size[0]), interpolation=cv2.INTER_NEAREST)
        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # h, w, c
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1
        
        ''' 3d gt'''
        ipm_gt_instance = torch.tensor(ipm_gt).unsqueeze(0)  # h, w, c0
        ipm_gt_offset = torch.tensor(offset_y_map).unsqueeze(0)
        ipm_gt_z = torch.tensor(z_map).unsqueeze(0)
        ipm_gt_segment = torch.clone(ipm_gt_instance)
        ipm_gt_segment[ipm_gt_segment > 0] = 1
        image_gt_heightmap = torch.tensor(bev_height_map_mask)

        return image, ipm_gt_segment.float(), ipm_gt_instance.float(), ipm_gt_offset.float(), ipm_gt_z.float(), image_gt_segment.float(), image_gt_instance.float(), intrinsic.float(), image_gt_heightmap.float()

    def __len__(self):
        return len(self.cnt_list)


if __name__ == "__main__":
    ''' parameter from config '''
    from utils.config_util import load_config_module
    config_file = '/mnt/ve_perception/wangruihao/code/BEV-LaneDet/tools/openlane_config.py'
    configs = load_config_module(config_file)
    dataset = configs.val_dataset()
    for item in dataset:
        continue