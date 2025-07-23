import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
import albumentations as A
from openlane_data import OpenLane_dataset_with_offset
from albumentations.pytorch import ToTensorV2

def precompute_bev_heightmaps(
    image_paths,
    gt_paths,
    map_paths,
    x_range,
    y_range,
    meter_per_pixel,
    data_trans,
    input_shape,
    output_2d_shape,
    save_dir
):
    """
    OpenLane_dataset_with_offset 을 사용하여,
    전체 sample 에 대해 get_seg_offset() 으로 BEV Heightmap을 미리 계산하고 저장한다.
    """
    # Dataset 인스턴스 생성
    dataset = OpenLane_dataset_with_offset(
        image_paths=image_paths,
        gt_paths=gt_paths,
        heightmap_paths=map_paths,
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        data_trans=data_trans,
        input_shape=input_shape,
        output_2d_shape=output_2d_shape
    )

    # 저장할 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Precomputing BEV Heightmaps"):
        # 현재 샘플에 해당하는 card, json 파일명
        card, cnt = dataset.cnt_list[idx]
        # 예: cnt == '000090.json' 같은 문자열

        # 카드별 폴더(= 이미지/gt 와 같은 구조) 생성
        card_folder = os.path.join(save_dir, card)
        os.makedirs(card_folder, exist_ok=True)

        # 저장할 파일명: 예) '000090.npy'
        bev_filename = cnt.replace('json', 'npy')
        bev_save_path = os.path.join(card_folder, bev_filename)

        # 이미 계산되어 있다면 넘어가도록(필요 시)
        if os.path.exists(bev_save_path):
            continue

        # get_seg_offset 호출 -> 필요한 부분만 가져옴
        # 만약 get_seg_offset 리턴이 아래와 같다면:
        #   image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic,
        #   bev_height_map_mask, matrix_lane2persformer
        # 라고 가정

        bev_height_map_mask= dataset.get_seg_offset_for_heightmap(idx)

        # numpy로 저장 (예: (2, H, W) shape)
        np.save(bev_save_path, bev_height_map_mask)

    print("=== Done precomputing BEV Heightmaps ===")

train_gt_paths = '/home/work/chase_data/openlane/validation'
train_image_paths = '/home/work/chase_data/openlane/images/validation'
train_map_paths = '/home/work/chase_data/openlane/map_data_validation'
train_heightmap_paths = '/home/work/chase_data/openlane/heightmap_validation'
input_shape = (600,800)

trans_image = A.Compose([
    A.Resize(height=input_shape[0], width=input_shape[1]),
    A.Normalize(),
    ToTensorV2()])
if __name__ == "__main__":
    ''' parameter from config '''
    precompute_bev_heightmaps(image_paths=train_image_paths,
                              gt_paths=train_gt_paths,
                              map_paths=train_map_paths,
                              x_range=(3, 103),
                              y_range=(-12, 12),
                              meter_per_pixel=0.5,
                              input_shape=input_shape,
                              data_trans=trans_image,
                              output_2d_shape=(144,256),
                              save_dir=train_heightmap_paths)