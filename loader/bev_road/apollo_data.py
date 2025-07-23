import copy
import json
import os
import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from utils.coord_util import ego2image,IPM2ego_matrix
from utils.standard_camera_cpu import Standard_camera

def extract_heightmap_within_range(point_cloud_vehicle, forward_range, lateral_range):
    ## point_cloud_vehicle shape: N,3 
    x = point_cloud_vehicle[:, 0] 
    y = point_cloud_vehicle[:, 1]
    
    # Filter points within the specified range
    mask = (x >= lateral_range[0]) & (x <= lateral_range[1]) & (y >=forward_range[0]) & (y <= forward_range[1])
    filtered_point_cloud = point_cloud_vehicle[mask]
    return filtered_point_cloud

def bev2ipm(bev, matrix_IPM2ego):
    ego_points = np.array([bev[0], bev[1]])
    ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (ego_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
    ipm_points_ = np.zeros_like(ipm_points)
    ipm_points_[0] = ipm_points[1]
    ipm_points_[1] = ipm_points[0]
    res_points = np.concatenate([ipm_points, np.array([bev[2]])], axis=0)
    return res_points 
def generate_bev_height_map(point_cloud, resolution=(200,48), meter_per_pixel=0.5, vehicle_height=0):
    bev_resolution_x, bev_resolution_y = resolution
        
    # Initialize BEV height map
    bev_height_map = np.zeros((bev_resolution_x, bev_resolution_y))
    # 각 픽셀에 대한 Z 값의 합과 개수를 저장할 배열 초기화
    z_sum_array = np.zeros((bev_resolution_x, bev_resolution_y))
    count_array = np.zeros((bev_resolution_x, bev_resolution_y))
    
    # 포인트 클라우드 반복하면서 각 픽셀에 대한 Z 값의 합과 개수 계산
    x_coords = point_cloud[:, 0].astype(int)
    y_coords = point_cloud[:, 1].astype(int)
    z_values = point_cloud[:, 2]
    
    # 유효한 픽셀 좌표 찾기
    valid_pixels_mask = (x_coords >= 0) & (x_coords < bev_resolution_x) & (y_coords >= 0) & (y_coords < bev_resolution_y)

    # 각 픽셀에 대한 Z 값의 합과 개수 계산
    np.add.at(z_sum_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), z_values[valid_pixels_mask])
    np.add.at(count_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), 1)

    # 계산된 Z 값의 합을 개수로 나누어 평균 계산
    with np.errstate(divide='ignore', invalid='ignore'):  # 0으로 나누는 오류 무시
        bev_height_map = np.divide(z_sum_array, count_array, out=np.zeros_like(z_sum_array), where=count_array != 0)
        
    #interpolated_height_map = griddata(known_points, known_values, (x_grid, y_grid), method='linear')
    return bev_height_map

def generate_binary_mask(bev_height_map):
    # Initialize binary mask
    binary_mask = np.zeros_like(bev_height_map)

    # Mark pixels with points as 1
    binary_mask[bev_height_map != 0] = 1

    return binary_mask

class Apollo_dataset_with_offset(Dataset):
    def __init__(self,data_json_path,
                 dataset_base_dir,
                 x_range,
                 y_range,
                 meter_per_pixel,
                 data_trans,
                 output_2d_shape):

        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.cnt_list = []
        self.lane3d_thick = 1
        self.lane2d_thick = 3
        json_file_path = data_json_path
        self.dataset_base_dir = dataset_base_dir

        self.except_list = ['06', '07', '08', '09', '10', '11']
        ''' virtual camera paramter'''
        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)
                name_list = info_dict['raw_file'].split('/')
                if name_list[-2] in self.except_list:
                    continue
                self.cnt_list.append(info_dict)
        print(len(self.cnt_list))
        ''' transform loader '''
        self.output2d_size = output_2d_shape
        self.trans_image = data_trans
        self.ipm_h, self.ipm_w = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
            (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)
        self.matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)

    def get_y_offset_and_z(self,res_d):
        def caculate_distance(base_points, lane_points, lane_z, lane_points_set):
            condition = np.where(
                (lane_points_set[0] == int(base_points[0])) & (lane_points_set[1] == int(base_points[1])))
            if len(condition[0]) == 0:
                return None,None
            lane_points_selected = lane_points.T[condition]  # 找到bin
            lane_z_selected = lane_z.T[condition]
            offset_y = np.mean(lane_points_selected[:, 1]) - base_points[1]
            z = np.mean(lane_z_selected[:, 1])
            return offset_y, z  # @distances.argmin(),distances[min_idx] #1#lane_points_selected[distances.argmin()],distances.min()

        # 画mask
        res_lane_points = {}
        res_lane_points_z = {}
        res_lane_points_bin = {}
        res_lane_points_set = {}
        for idx in res_d:
            ipm_points_ = np.array(res_d[idx])
            ipm_points = ipm_points_.T[np.where((ipm_points_[1] >= 0) & (ipm_points_[1] < self.ipm_h))].T  # 进行筛选
            if len(ipm_points[0]) <= 1:
                continue
            x, y, z = ipm_points[1], ipm_points[0], ipm_points[2]
            base_points = np.linspace(x.min(), x.max(),
                                      int((x.max() - x.min()) // 0.05))  # 画 offset 用得 画的非常细 一个格子里面20个点
            base_points_bin = np.linspace(int(x.min()), int(x.max()),
                                          int(int(x.max()) - int(x.min()))+1)  # .astype(np.int)
            # print(len(x),len(y),len(y))
            if len(x) == len(set(x)):
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(x, y, kind='linear',
                                         fill_value="extrapolate")  # 线性插值 #三次样条插值 kind='quadratic' linear cubic
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')
            else:
                sorted_index = np.argsort(x)[::-1] # 从大到小
                x_,y_,z_ = [],[],[]
                for x_index in range(len(sorted_index)): # 越来越小
                    if x[sorted_index[x_index]] >= x[sorted_index[x_index-1]] and x_index !=0:
                        continue
                    else:
                        x_.append(x[sorted_index[x_index]])
                        y_.append(y[sorted_index[x_index]])
                        z_.append(z[sorted_index[x_index]])
                x,y,z = np.array(x_),np.array(y_),np.array(z_)
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(x, y, kind='linear',
                                         fill_value="extrapolate")  # 线性插值 #三次样条插值 kind='quadratic' linear cubic
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')

            y_points = function1(base_points)
            y_points_bin = function1(base_points_bin)
            z_points = function2(base_points)
            # cv2.polylines(instance_seg, [ipm_points.T.astype(np.int)], False, idx+1, 1)
            res_lane_points[idx] = np.array([base_points, y_points])  # 
            res_lane_points_z[idx] = np.array([base_points, z_points])
            res_lane_points_bin[idx] = np.array([base_points_bin, y_points_bin]).astype(np.int)  # 画bin用的
            res_lane_points_set[idx] = np.array([base_points, y_points]).astype(
                np.int)  
        offset_map = np.zeros((self.ipm_h, self.ipm_w))
        z_map = np.zeros((self.ipm_h, self.ipm_w))
        ipm_image = np.zeros((self.ipm_h, self.ipm_w))
        for idx in res_lane_points_bin:
            lane_bin = res_lane_points_bin[idx].T
            for point in lane_bin:
                row, col = point[0], point[1]
                if not ( 0 < row < self.ipm_h and 0 < col < self.ipm_w): # 没有在视野内部的去除掉
                    continue
                ipm_image[row, col] = idx
                center = np.array([row, col])
                offset_y, z = caculate_distance(center, res_lane_points[idx], res_lane_points_z[idx],
                                                res_lane_points_set[idx])  # 根据距离选idex
                if offset_y is None: #
                    ipm_image[row,col] = 0
                    continue
                if offset_y > 1:
                    print('haha')
                    offset_y = 1
                if offset_y < 0:
                    print('hahahahahha')
                    offset_y = 0
                offset_map[row][col] = offset_y
                z_map[row][col] = z

        return ipm_image,offset_map,z_map


    def get_seg_offset(self,idx):
        info_dict = self.cnt_list[idx]
        name_list = info_dict['raw_file'].split('/')
    
        image_path = os.path.join(self.dataset_base_dir, 'images', name_list[-2], name_list[-1])
        map_path = os.path.join(self.dataset_base_dir, 'map', name_list[-2], name_list[-1].replace("jpg", 'npy'))
        
        image = cv2.imread(image_path)
        pointsxyz = np.load(map_path)
        
        # caculate camera parameter
        cam_height, cam_pitch = info_dict['cam_height'], info_dict['cam_pitch']
        project_g2c, camera_k = self.get_camera_matrix(cam_pitch, cam_height)
        project_c2g = np.linalg.inv(project_g2c)

        roi_pointcloud = extract_heightmap_within_range(pointsxyz, self.x_range, self.y_range)
        roi_pointcloud = np.vstack((roi_pointcloud.T, np.ones((1, roi_pointcloud.shape[0]))))
        
        roi = np.array([roi_pointcloud[1], -roi_pointcloud[0], roi_pointcloud[2]])
        ipm_roi = bev2ipm(roi, self.matrix_IPM2ego)
        
        bev_heightmap = generate_bev_height_map(ipm_roi.T, resolution=(self.ipm_h, self.ipm_w), meter_per_pixel=self.meter_per_pixel)
        bev_heightmask = generate_binary_mask(bev_heightmap)
        bev_height_map_mask = np.stack((bev_heightmap, bev_heightmask), axis=0)
        
        # caculate point
        lane_grounds = info_dict['laneLines']
        image_gt = np.zeros(image.shape[:2], dtype=np.uint8)
        res_points_d = {}
        for lane_idx in range(len(lane_grounds)):
            # select point by visibility
            lane_visibility = np.array(info_dict['laneLines_visibility'][lane_idx])
            lane_ground = np.array(lane_grounds[lane_idx])
            assert lane_visibility.shape[0] == lane_ground.shape[0]
            lane_ground = lane_ground[lane_visibility > 0.5]
            lane_ground = np.concatenate([lane_ground, np.ones([lane_ground.shape[0], 1])], axis=1).T
            # get image gt
            lane_camera = np.matmul(project_g2c, lane_ground)
            lane_image = camera_k @ lane_camera[:3]
            lane_image = lane_image / lane_image[2]
            lane_uv = lane_image[:2].T
            cv2.polylines(image_gt, [lane_uv.astype(np.int)], False, lane_idx + 1, 3)
            x, y, z = lane_ground[1], -1 * lane_ground[0], lane_ground[2]
            ground_points = np.array([x, y])
            ipm_points = np.linalg.inv(self.matrix_IPM2ego[:, :2]) @ (
                        ground_points[:2] - self.matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)
            res_points_d[lane_idx+1] = res_points

        bev_gt,offset_y_map,z_map = self.get_y_offset_and_z(res_points_d)
        return image,image_gt,bev_gt,offset_y_map,z_map,project_g2c,camera_k, bev_height_map_mask


    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        image, image_gt, bev_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, bev_height_map_mask = self.get_seg_offset(idx)
        image_h, image_w, _ = image.shape
        cam_intrinsic[0] *= 800 / image_w
        cam_intrinsic[1] *= 600 / image_h
        transformed = self.trans_image(image=image)
        intrinsic = torch.tensor(cam_intrinsic)
        extrinsic = torch.tensor(cam_extrinsics)
        road2cam = torch.tensor(cam_extrinsics)
        image = transformed["image"]
        ''' 2d gt'''
        image_gt = cv2.resize(image_gt, (self.output2d_size[1],self.output2d_size[0]), interpolation=cv2.INTER_NEAREST)
        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # h, w, c
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1
        ''' 3d gt'''
        bev_gt_instance = torch.tensor(bev_gt).unsqueeze(0)  # h, w, c0
        bev_gt_offset = torch.tensor(offset_y_map).unsqueeze(0)
        bev_gt_z = torch.tensor(z_map).unsqueeze(0)
        bev_gt_segment = torch.clone(bev_gt_instance)
        bev_gt_segment[bev_gt_segment > 0] = 1
        gt_heightmap = torch.tensor(bev_height_map_mask)
        return image, bev_gt_segment.float(), bev_gt_instance.float(),bev_gt_offset.float(),bev_gt_z.float(),image_gt_segment.float(),image_gt_instance.float(),intrinsic.float(), extrinsic.float(), road2cam.float(), gt_heightmap.float()


    def get_camera_matrix(self,cam_pitch,cam_height):
        proj_g2c = np.array([[1,                             0,                              0,          0],
                            [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                            [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                            [0,                             0,                              0,          1]])


        camera_K = np.array([[2015., 0., 960.],
                        [0., 2015., 540.],
                        [0., 0., 1.]])


        return proj_g2c,camera_K

    def __len__(self):
        return len(self.cnt_list)


class Apollo_dataset_with_offset_val(Dataset):
    def __init__(self,data_json_path,
                 dataset_base_dir,
                 data_trans):
        self.except_list = ['06', '07', '08', '09', '10', '11']
        self.cnt_list = []
        json_file_path = data_json_path
        self.dataset_base_dir = dataset_base_dir
        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)
                name_list = info_dict['raw_file'].split('/')
                # if name_list[-2] in self.except_list:
                    # continue
                self.cnt_list.append(info_dict)
                
        print(len(self.cnt_list))
        ''' transform loader '''
        self.trans_image = data_trans


    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        
        info_dict = self.cnt_list[idx]
        name_list = info_dict['raw_file'].split('/')
                
        image_path = os.path.join(self.dataset_base_dir, 'images', name_list[-2], name_list[-1])
        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape
        
        # caculate camera parameter
        cam_height, cam_pitch = info_dict['cam_height'], info_dict['cam_pitch']
        project_g2c, camera_k = self.get_camera_matrix(cam_pitch, cam_height)
        project_c2g = np.linalg.inv(project_g2c)
        
        camera_k[0] *= 800 / image_w
        camera_k[1] *= 600 / image_h
        transformed = self.trans_image(image=image)
        image = transformed["image"]
        return image,camera_k, project_g2c, name_list[1:]
    

    
    def get_camera_matrix(self,cam_pitch,cam_height):
        proj_g2c = np.array([[1,                             0,                              0,          0],
                            [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                            [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                            [0,                             0,                              0,          1]])


        camera_K = np.array([[2015., 0., 960.],
                        [0., 2015., 540.],
                        [0., 0., 1.]])


        return proj_g2c,camera_K

    def __len__(self):
        return len(self.cnt_list)


if __name__ == '__main__':
    ''' parameter from config '''
    from utils.config_util import load_config_module
    config_file = '/mnt/ve_perception/wangruihao/code/BEV-LaneDet/tools/apollo_config.py'
    configs = load_config_module(config_file)
    dataset = configs.val_dataset()
    for item in dataset:
        continue
