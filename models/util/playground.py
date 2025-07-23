ret = self.collect_nd_embedding_with_position(seg, embedding, self.post_conf)
c = self.naive_cluster_nd(ret, self.post_emb_margin)

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



ret = self.collect_embedding_with_position(seg, embedding, self.post_conf)
c = self.naive_cluster(ret, self.post_emb_margin)

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



def ego2image(self, ego_points, camera_intrinsic, ego2camera_matrix, output_2d =(144,256)):
    camera_points = torch.matmul(ego2camera_matrix[:3, :3], ego_points) + \
                ego2camera_matrix[:3, 3].unsqueeze(1)
    image_points_ = torch.matmul(camera_intrinsic, camera_points)
    image_points = image_points_ / image_points_[2, :]
    mask = (image_points[0] >= 0) & (image_points[0] < output_2d[1]) & \
        (image_points[1] >= 0) & (image_points[1] < output_2d[0])
    image_points = image_points[:, mask]
    return image_points


def projection2d(self, batch_lines, intrinsics, extrinsic):
    b = len(batch_lines)
    pred_2d = torch.zeros((b,144,256), device='cuda:0')
    for b,lines in enumerate(batch_lines):
        intrinsic = intrinsics[b]
        intrinsic[0] *= 256 / self.image_w
        intrinsic[1] *= 144 / self.image_h      
            
        for x,y,z in lines:
            lane = torch.stack([x,y,z], dim=0)
            uv1 = self.ego2image(lane, intrinsic, extrinsic[b], output_2d = (144,256))
            x = uv1[0,:].long()
            y = uv1[1,:].long()
            pred_2d[b,y,x] = 1

    return pred_2d