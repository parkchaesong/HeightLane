import json
import pdb
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


def bev_to_image_projection(bev_lanes, offsets, heightmaps, intrinsics, extrinsics, image_shape, output_2d = (256, 144)):
    """
    BEV lane to image projection using heightmap.
    
    :param bev_lane: BEV canvas with lanes (torch.Tensor) shape:  [batch, 1, height, width]
    :param offsets: Offsets for the BEV canvas (torch.Tensor) shape: [batch, 1, height, width]
    :param heightmap: Heightmap for the BEV canvas (torch.Tensor) shape: [batch, 1, height, width]
    :param intrinsic: Camera intrinsic matrix (torch.Tensor) shape: [batch, 3, 3]
    :param extrinsic: Camera extrinsic matrix (torch.Tensor) shape: [batch, 4, 4]
    :param image_shape: Tuple containing image height and width (height, width)
    :return: Image with projected lanes (torch.Tensor) shape: [batch, height, width]
    """
    # BEV canvas size
    batch_size = bev_lanes.shape[0]
    img_height, img_width = image_shape
    output_images = torch.zeros((batch_size, output_2d[1], output_2d[0]), dtype=torch.float32, device=bev_lanes.device)
    maxtrix_cameraw2camera = torch.tensor(np.array(([[-0., -1., -0., -0.],
                                        [-0., -0., -1., -0.],
                                        [ 1.,  0.,  0.,  0.],
                                        [ 0.,  0.,  0.,  1.]])), dtype=torch.float32).to(intrinsics.device)

    # Create a grid of BEV coordinates
    x = torch.linspace(0, bev_lanes.shape[2] - 1, bev_lanes.shape[2], device=bev_lanes.device)
    y = torch.linspace(0, bev_lanes.shape[3] - 1, bev_lanes.shape[3], device=bev_lanes.device)
    xv_, yv_ = torch.meshgrid(x, y)
    for b in range(batch_size):
        intrinsic = intrinsics[b]
        extrinsic = extrinsics[b]

        inv_np = maxtrix_cameraw2camera @ torch.linalg.inv(extrinsic)
        extrinsic = inv_np.to(intrinsic.device).type(intrinsic.dtype)

        lane_mask = bev_lanes[b][0] > -1e4
        indices = torch.nonzero(lane_mask, as_tuple=True)
        offsetv = offsets[b][0][indices]
        xv = 103 -((xv_[indices] +0.5) / 2) 
        yv = (yv_[indices] / 2) - 12 + offsetv/ 2
        zv = heightmaps[b][0][indices]

        # Convert BEV coordinates to world coordinates
        bev_coords = torch.stack([yv, xv, zv], dim=0)  # shape: [3, num_lane_pixels]
        bev_coords = bev_coords.view(3, -1)  # shape: [3, num_lane_pixels]

        # Transform BEV coordinates to camera coordinates
        bev_coords = torch.cat([bev_coords, torch.ones(1, bev_coords.shape[1], device=bev_lanes.device)], dim=0)  # shape: [4, num_lane_pixels]
        cam_coords = torch.matmul(extrinsic, bev_coords)  # shape: [3, num_lane_pixels]

        # Adjust intrinsic matrix for output image size
        intrinsic[0] *= output_2d[0] / img_width
        intrinsic[1] *= output_2d[1] / img_height
        # Project camera coordinates to image plane
        img_coords = torch.matmul(intrinsic, cam_coords[:3, :])  # shape: [3, num_lane_pixels]
        img_coords = img_coords[:2, :] / img_coords[2, :]  # Normalize by z to get pixel coordinates

        # Filter out coordinates that are outside the image bounds
        mask = (img_coords[0] >= 0) & (img_coords[0] < output_2d[0]) & (img_coords[1] >= 0) & (img_coords[1] <  output_2d[1])
        # pdb.set_trace()
        # Get valid pixel coordinates
        # img_coords = img_coords[:,mask].long()
        img_x = img_coords[0, mask].long()
        img_y = img_coords[1, mask].long()

        # Map BEV lane values to the image
        output_images[b, img_y, img_x] = bev_lanes[b][0][indices][mask]
        # output_2d_coord.append(img_coords)
    return output_images

def mean_col_by_row_with_offset_z(seg, offset_y, z):
    assert (len(seg.shape) == 2)

    center_ids = np.unique(seg[seg > 0])
    lines = []
    for idx, cid in enumerate(center_ids):  # 一个id
        cols, rows, z_val = [], [], []
        for y_op in range(seg.shape[0]):  # Every row
            condition = seg[y_op, :] == cid
            x_op = np.where(condition)[0]  # All pos in this row
            z_op = z[y_op, :]
            offset_op = offset_y[y_op, :]
            if x_op.size > 0:
                offset_op = offset_op[x_op]
                z_op = np.mean(z_op[x_op])
                z_val.append(z_op)
                x_op_with_offset = x_op + offset_op
                x_op = np.mean(x_op_with_offset)  # mean pos
                cols.append(x_op)
                rows.append(y_op)
        lines.append((cols, rows, z_val))
    return lines



def bev_instance2points_with_offset_z(ids: np.ndarray, max_x=50, meter_per_pixal=(0.2, 0.2), offset_y=None, Z=None):
    center = ids.shape[1] / 2
    lines = mean_col_by_row_with_offset_z(ids, offset_y, Z)
    points = []
    # for i in range(1, ids.max()):
    for y, x, z in lines:  # cols, rows
        # x, y = np.where(ids == 1)
        x = np.array(x)[::-1]
        y = np.array(y)[::-1]
        z = np.array(z)[::-1]

        x = max_x / meter_per_pixal[0] - x
        y = y * meter_per_pixal[1]
        y -= center * meter_per_pixal[1]
        x = x * meter_per_pixal[0]

        y *= -1.0  # Vector is from right to left
        if len(x) < 2:
            continue
        spline = CubicSpline(x, y, extrapolate=False)
        points.append((x, y, z, spline))
    return points


class PostProcessingModule(torch.nn.Module):
    def __init__(self, post_conf, post_emb_margin, post_min_cluster_size, max_x, meter_per_pixel, image_shape):
        super().__init__()
        self.post_conf = post_conf
        self.post_emb_margin = post_emb_margin
        self.post_min_cluster_size = post_min_cluster_size
        self.max_x = max_x
        self.meter_per_pixel = meter_per_pixel
        self.image_h, self.image_w = image_shape

    def forward(self, pred_, intrinsic, extrinsic, projection=False):
        seg = pred_[0]  # [batch_size, 1, 200, 48]
        embedding = pred_[1] # [batch_size, num_channels, 200, 48]
        offset_y = torch.sigmoid(pred_[2])  # [batch_size, 1, 200, 48]
        z_pred = pred_[3]  # [batch_size, 1, 200, 48]
        b, nd, h, w = embedding.shape
        if nd > 1:
            # c = self.collect_and_cluster_nd(seg, embedding, self.post_conf, self.post_emb_margin)
            ret = self.collect_nd_embedding_with_position(seg, embedding, self.post_conf)
            c = self.naive_cluster_nd(ret, self.post_emb_margin)
        else:
            c = self.collect_and_cluster(seg, embedding, self.post_conf, self.post_emb_margin)
            # ret = self.collect_embedding_with_position(seg, embedding, self.post_conf)
            # c = self.naive_cluster(ret, self.post_emb_margin)

        lanes = torch.zeros_like(seg, dtype=torch.uint8, device='cuda:0')
        for b, cids in enumerate(c[0]):
            for x, y, id in cids:
                if c[1][b][id][1] < self.post_min_cluster_size:
                    continue
                lanes[b, 0, x, y] = id + 1

        batch_lines = self.mean_col_by_row_with_offset_z_batch(lanes, offset_y, z_pred)
        # print(batch_lines[0])
        batch_points = []
        for lines in batch_lines:
            points = []
            for y, x, z in lines:  # cols, rows
                # Reverse the tensors
                x = x.flip(dims=[0])
                y = y.flip(dims=[0])
                z = z.flip(dims=[0])

                # Perform the calculations
                x = self.max_x / self.meter_per_pixel[0] - x
                y = y * self.meter_per_pixel[1]
                y -= (w // 2) * self.meter_per_pixel[1]
                x = x * self.meter_per_pixel[0]

                # Invert y-axis
                y = -y

                # Check for sufficient points for spline
                if len(x) >= 2:
                    points.append((x, y, z))

            batch_points.append(points)
        if projection:
            batched_image = self.projection2d(batch_points, intrinsic, extrinsic)
            return batch_points, batched_image
        return batch_points
    def collect_nd_embedding_with_position(self, seg, emb, conf):
        batch_ret = []
        for b in range(seg.shape[0]):
            # 현재 배치의 세그먼트 데이터
            current_seg = seg[b, 0]  # shape: [height, width]

            # 현재 배치의 임베딩 데이터
            current_emb = emb[b]  # shape: [num_features, height, width]

            # conf 이상의 값에 대한 마스크 생성
            mask = current_seg >= conf

            # 마스크에서 True인 위치의 인덱스를 추출
            i_indices, j_indices = torch.where(mask)

            # 해당 인덱스를 사용하여 현재 배치에서 임베딩 추출
            embeddings = current_emb[:, i_indices, j_indices]

            # 현재 배치의 결과 리스트 생성
            ret = [(i.item(), j.item(), emb_tensor) for i, j, emb_tensor in zip(i_indices, j_indices, embeddings.t())]
            batch_ret.append(ret)
        return batch_ret


    def naive_cluster_nd(self, batch_ret, emb_margin):
        batch_centers = []
        batch_cids = []
        for ret in batch_ret:
            cids = []
            centers = []
            for x, y, emb in ret:
                emb = torch.tensor(emb)
                if len(centers) == 0:
                    centers.append((emb, 1))
                    cids.append((x, y, 0))
                    continue
                
                # Calculate distances to all current centers
                center_tensors, counts = zip(*centers)
                center_tensors = torch.stack(center_tensors)
                distances = torch.norm(center_tensors - emb, dim=1)

                # Find the closest center
                min_dist, min_cid = torch.min(distances, dim=0)
                if min_dist < emb_margin:
                    # Update center and count
                    center_count = counts[min_cid]
                    new_center = (center_tensors[min_cid] * center_count + emb) / (center_count + 1)
                    centers[min_cid] = (new_center, center_count + 1)
                    cids.append((x, y, min_cid))
                else:
                    centers.append((emb, 1))
                    cids.append((x, y, len(centers) - 1))

            batch_centers.append(centers)
            batch_cids.append(cids)
        return batch_cids, batch_centers
    def naive_cluster(self, batch_ret, emb_margin):
        batch_centers = []
        batch_cids = []
        for ret in batch_ret:
            cids = []
            centers = []
            for x, y, emb in ret:
                emb = torch.tensor(emb)
                if len(centers) == 0:
                    centers.append((emb, 1))
                    cids.append((x, y, 0))
                    continue
                
                # Calculate distances to all current centers
                center_tensors, counts = zip(*centers)
                center_tensors = torch.stack(center_tensors)
                distances = torch.norm(center_tensors - emb)

                # Find the closest center
                min_dist, min_cid = torch.min(distances, dim=0)
                if min_dist < emb_margin:
                    # Update center and count
                    center_count = counts[min_cid]
                    new_center = (center_tensors[min_cid] * center_count + emb) / (center_count + 1)
                    centers[min_cid] = (new_center, center_count + 1)
                    cids.append((x, y, min_cid))
                else:
                    centers.append((emb, 1))
                    cids.append((x, y, len(centers) - 1))

            batch_centers.append(centers)
            batch_cids.append(cids)
        return batch_cids, batch_centers

    def mean_col_by_row_with_offset_z_batch(self, seg, offset_y_batch, z_batch):
        assert (len(seg.shape) == 4)
        assert (len(offset_y_batch.shape) == 4)
        batch_size = seg.shape[0]
        batch_lines = []
        for b in range(batch_size):
            ids = seg[b, 0]
            offset_y = offset_y_batch[b, 0]
            z = z_batch[b, 0]

            # Extract unique IDs, ignoring zero (background)
            center_ids = torch.unique(ids[ids > 0])

            lines = []
            for cid in center_ids:
                # Create masks and operations for the entire batch of `cid`
                mask = ids == cid
                x_indices = torch.nonzero(mask, as_tuple=True)
                y_indices, x_values = x_indices

                if y_indices.numel() == 0:
                    continue

                # Get offset and z values where mask is true
                offset_values = offset_y[mask]
                z_values = z[mask]

                # Adjust x coordinates and calculate mean values per unique y_index
                x_adjusted = x_values.float() + offset_values

                # Calculate mean x, y, z per unique y_index
                unique_rows, inverse_indices = torch.unique_consecutive(y_indices, return_inverse=True)
                mean_x_per_row = torch.zeros_like(unique_rows).float()
                mean_z_per_row = torch.zeros_like(unique_rows).float()
                counts_per_row = torch.bincount(inverse_indices, minlength=len(unique_rows))

                mean_x_per_row.index_add_(0, inverse_indices, x_adjusted)
                mean_z_per_row.index_add_(0, inverse_indices, z_values)

                mean_x_per_row /= counts_per_row.float()
                mean_z_per_row /= counts_per_row.float()
                rows_adjusted = unique_rows.float() + 0.5
 
                # Collect results as tensors for each cid
                lines.append((mean_x_per_row, rows_adjusted, mean_z_per_row))

            batch_lines.append(lines)

        return batch_lines
    
    def collect_embedding_with_position(self, seg, emb, conf):
        batch_ret = []

        # 전체 배치에 대해 벡터화된 마스크 생성
        masks = seg[:, 0] >= conf  # shape: [batch_size, height, width]

        for b in range(seg.shape[0]):
            # 현재 배치의 세그먼트 데이터와 마스크
            current_mask = masks[b]  # shape: [height, width]

            # 현재 배치의 임베딩 데이터
            current_emb = emb[b, 0]  # shape: [num_features, height, width]

            # 마스크에서 True인 위치의 인덱스를 추출
            i_indices, j_indices = torch.where(current_mask)

            # 해당 인덱스를 사용하여 현재 배치에서 임베딩 추출
            embeddings = current_emb[i_indices, j_indices]  # shape: [num_selected, num_features]

            # 현재 배치의 결과 리스트 생성
            ret = [(i.item(), j.item(), emb_tensor) for i, j, emb_tensor in zip(i_indices, j_indices, embeddings)]
            batch_ret.append(ret)
        
        return batch_ret
    
    def ego2image(self, ego_points, camera_intrinsic, ego2camera_matrix, output_2d =(144,256)):
        camera_points = torch.matmul(ego2camera_matrix[:3, :3], ego_points) + \
                    ego2camera_matrix[:3, 3].unsqueeze(1)
        image_points_ = torch.matmul(camera_intrinsic, camera_points)
        image_points = image_points_ / image_points_[2, :]
        mask = (image_points[0] >= 0) & (image_points[0] < output_2d[1]) & \
           (image_points[1] >= 0) & (image_points[1] < output_2d[0])
        image_points = image_points[:, mask]
        return image_points

    
    def projection2d(self, batch_lines, intrinsics, extrinsic, output_2d=(144, 256)):
            b = len(batch_lines)
            pred_2d = torch.zeros((b, output_2d[0], output_2d[1]), device='cuda:0')

            for batch_idx, lines in enumerate(batch_lines):
                intrinsic = intrinsics[batch_idx].clone()
                intrinsic[0] *= output_2d[1] / self.image_w
                intrinsic[1] *= output_2d[0] / self.image_h

                all_lanes = []
                for x, y, z in lines:
                    lane = torch.stack([x, y, z], dim=0)
                    all_lanes.append(lane)

                if not all_lanes:
                    continue

                all_lanes = torch.cat(all_lanes, dim=1)
                image_points = self.ego2image(all_lanes, intrinsic, extrinsic[batch_idx])

                # Convert to integer pixel values
                x = image_points[0, :].long()
                y = image_points[1, :].long()

                # Set pixel values in the prediction tensor
                pred_2d[batch_idx, y, x] = 1

            return pred_2d
    
    def collect_and_cluster_nd(self, seg, embedding, conf, emb_margin):
        batch_centers = []
        batch_cids = []

        for b in range(seg.shape[0]):
            # 현재 배치의 세그먼트 데이터
            current_seg = seg[b, 0]  # shape: [height, width]

            # 현재 배치의 임베딩 데이터
            current_emb = embedding[b]  # shape: [num_features, height, width]

            # conf 이상의 값에 대한 마스크 생성
            mask = current_seg >= conf

            # 마스크에서 True인 위치의 인덱스를 추출
            i_indices, j_indices = torch.where(mask)

            if len(i_indices) == 0:
                batch_centers.append([])
                batch_cids.append([])
                continue

            # 해당 인덱스를 사용하여 현재 배치에서 임베딩 추출
            embeddings = current_emb[:, i_indices, j_indices].t()  # shape: [num_selected, num_features]

            # 현재 배치의 결과 리스트 생성
            cids = torch.full((len(i_indices),), -1, dtype=torch.int64)  # 초기 cluster ids
            centers = []

            for idx in range(len(i_indices)):
                emb = embeddings[idx]
                if len(centers) == 0:
                    centers.append((emb, 1))
                    cids[idx] = 0
                    continue

                # Calculate distances to all current centers
                center_tensors, counts = zip(*centers)
                center_tensors = torch.stack(center_tensors)
                distances = torch.norm(center_tensors - emb, dim=1)

                # Find the closest center
                min_dist, min_cid = torch.min(distances, dim=0)
                if min_dist < emb_margin:
                    # Update center and count
                    center_count = counts[min_cid]
                    new_center = (center_tensors[min_cid] * center_count + emb) / (center_count + 1)
                    centers[min_cid] = (new_center, center_count + 1)
                    cids[idx] = min_cid.item()
                else:
                    centers.append((emb, 1))
                    cids[idx] = len(centers) - 1

            batch_centers.append(centers)
            batch_cids.append([(i.item(), j.item(), cid.item()) for i, j, cid in zip(i_indices, j_indices, cids)])

        return batch_cids, batch_centers
    
    
    def collect_and_cluster(self, seg, emb, conf, emb_margin):
        batch_cids = []
        batch_centers = []
        masks = seg[:, 0] >= conf  # shape: [batch_size, height, width]
        for b in range(seg.shape[0]):
            # 현재 배치의 세그먼트 데이터와 마스크
            current_mask = masks[b]  # shape: [height, width]

            # 현재 배치의 임베딩 데이터
            current_emb = emb[b,0]  # shape: [num_features, height, width]

            # 마스크에서 True인 위치의 인덱스를 추출
            i_indices, j_indices = torch.where(current_mask)

            if len(i_indices) == 0:
                batch_centers.append([])
                batch_cids.append([])
                continue

            # 해당 인덱스를 사용하여 현재 배치에서 임베딩 추출
            embeddings = current_emb[i_indices, j_indices]  # shape: [num_selected, num_features]

            # 현재 배치의 클러스터링 결과 생성
            cluster_ids = torch.full((len(i_indices),), -1, dtype=torch.int64)  # 초기 cluster ids
            centers = []

            for idx in range(len(i_indices)):
                emb_vector = embeddings[idx]
                if len(centers) == 0:
                    centers.append((emb_vector, 1))
                    cluster_ids[idx] = 0
                    continue

                # 모든 현재 센터들에 대한 거리 계산
                center_tensors, counts = zip(*centers)
                center_tensors = torch.stack(center_tensors)
                distances = torch.norm(center_tensors - emb_vector)

                # 가장 가까운 센터 찾기
                min_dist, min_cid = torch.min(distances, dim=0)
                if min_dist < emb_margin:
                    # 센터와 카운트 업데이트
                    center_count = counts[min_cid]
                    new_center = (center_tensors[min_cid] * center_count + emb_vector) / (center_count + 1)
                    centers[min_cid] = (new_center, center_count + 1)
                    cluster_ids[idx] = min_cid.item()
                else:
                    centers.append((emb_vector, 1))
                    cluster_ids[idx] = len(centers) - 1

            batch_centers.append(centers)
            batch_cids.append([(i.item(), j.item(), cid.item()) for i, j, cid in zip(i_indices, j_indices, cluster_ids)])

        return batch_cids, batch_centers