import numpy as np
# from utils.utils import *
from utils.util_val.MinCostFlow import SolveMinCostFlow
from utils.util_val.utils import *
from pprint import pprint


class LaneEval(object):
    def __init__(self):
        '''
            args.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
            args.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
        '''

        self.x_min = -10
        self.x_max = 10
        self.y_min = 3
        self.y_max = 103
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40
        self.laneline_x_error_close = []
        self.laneline_x_error_far = []
        self.laneline_z_error_close = []
        self.laneline_z_error_far = []
        self.r_list = []
        self.p_list =[]
        self.cnt_gt_list = []
        self.cnt_pred_list = []

    def bench(self, pred_lanes, gt_lanes):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param raw_file: file path rooted in dataset folder
        :param gt_cam_height: camera height given in ground-truth loader
        :param gt_cam_pitch: camera pitch given in ground-truth loader
        :return:
        """
        # change this properly
        close_range_idx = np.where(self.y_samples > self.close_range)[0][0]

        r_lane, p_lane, c_lane = 0., 0., 0.
        x_error_close = []
        x_error_far = []
        z_error_close = []
        z_error_far = []

        # only keep the visible portion
        gt_lanes = [np.array(gt_lane) for k, gt_lane in
                    enumerate(gt_lanes)]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]
        
        pred_lanes = [lane for lane in pred_lanes if np.array(lane)[0, 1] < self.y_samples[-1] and np.array(lane)[-1, 1] > self.y_samples[0]]
        pred_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in pred_lanes]
        pred_lanes = [lane for lane in pred_lanes if np.array(lane).shape[0] > 1]

        # only consider those gt lanes overlapping with sampling range 有交集的部分
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [prune_3d_lane_by_range(np.array(lane),  self.x_min, self.x_max) for lane in gt_lanes]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples, out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                     np.logical_and(self.y_samples >= min_y, self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples, out_vis=True)
            smoothed_z_values = np.convolve(z_values, np.ones(3)/3, mode='same')
            pred_lanes[i] = np.vstack([x_values, smoothed_z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                       np.logical_and(self.y_samples >= min_y, self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)
            # pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, x_values <= self.x_max)

        gt_lanes = [gt_lanes[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_visibility_mat = gt_visibility_mat[np.sum(gt_visibility_mat, axis=-1) > 1, :]
        cnt_gt = len(gt_lanes)


        pred_lanes = [pred_lanes[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_visibility_mat = pred_visibility_mat[np.sum(pred_visibility_mat, axis=-1) > 1, :]
        cnt_pred = len(pred_lanes)
        
        
        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close.fill(1000.)
        x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_far.fill(1000.)
        z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_close.fill(1000.)
        z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_far.fill(1000.)

        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred): #
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])

                # apply visibility to penalize different partial matching accordingly
                both_visible_indices = np.logical_and(gt_visibility_mat[i, :] >= 0.5, pred_visibility_mat[j, :] >= 0.5)
                both_invisible_indices = np.logical_and(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)
                other_indices = np.logical_not(np.logical_or(both_visible_indices, both_invisible_indices))

                euclidean_dist = np.sqrt(x_dist ** 2 + z_dist ** 2)
                euclidean_dist[both_invisible_indices] = 0
                euclidean_dist[other_indices] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th) - np.sum(both_invisible_indices)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # using num_match_mat as cost does not work?
                cost_ = np.sum(euclidean_dist)

                if cost_<1 and cost_>0:

                    cost_ = 1

                else:

                    cost_ = (cost_).astype(int)

                cost_mat[i, j] = cost_
                # cost_mat[i, j] = num_match_mat[i, j]

                # use the both visible portion to calculate distance error
                # both_visible_indices = np.logical_and(gt_visibility_mat[i, :] > 0.5, pred_visibility_mat[j, :] > 0.5)
                if np.sum(both_visible_indices[:close_range_idx]) > 0:
                    x_dist_mat_close[i, j] = np.sum(
                        x_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                    z_dist_mat_close[i, j] = np.sum(
                        z_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                else:
                    x_dist_mat_close[i, j] = -1
                    z_dist_mat_close[i, j] = -1

                if np.sum(both_visible_indices[close_range_idx:]) > 0:
                    x_dist_mat_far[i, j] = np.sum(
                        x_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                    z_dist_mat_far[i, j] = np.sum(
                        z_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                else:
                    x_dist_mat_far[i, j] = -1
                    z_dist_mat_far[i, j] =-1

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        match_num = 0
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.y_samples.shape[0]:
                    match_num += 1
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)
                    # if pred_category != []:
                    #     if pred_category[pred_i] == gt_category[gt_i] or (pred_category[pred_i]==20 and gt_category[gt_i]==21):
                    #         c_lane += 1    # category matched num
                    x_error_close.append(x_dist_mat_close[gt_i, pred_i])
                    x_error_far.append(x_dist_mat_far[gt_i, pred_i])
                    z_error_close.append(z_dist_mat_close[gt_i, pred_i])
                    z_error_far.append(z_dist_mat_far[gt_i, pred_i])
        return r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far

    def bench_all(self,pred_lanes,gt_lanes):
        r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, \
        x_error_close, x_error_far, \
        z_error_close, z_error_far = self.bench(pred_lanes,
                                                gt_lanes)
        # laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
        # consider x_error z_error only for the matched lanes
        # if r_lane > 0 and p_lane > 0:
        
        self.r_list.append(r_lane)
        self.p_list.append(p_lane)
        self.cnt_gt_list.append(cnt_gt)
        self.cnt_pred_list.append(cnt_pred)
        self.laneline_x_error_close.extend(x_error_close)
        self.laneline_x_error_far.extend(x_error_far)
        self.laneline_z_error_close.extend(z_error_close)
        self.laneline_z_error_far.extend(z_error_far)
        
        

    def show(self):
        laneline_x_error_close = np.array(self.laneline_x_error_close)
        laneline_x_error_far = np.array(self.laneline_x_error_far)
        laneline_z_error_close = np.array(self.laneline_z_error_close)
        laneline_z_error_far = np.array(self.laneline_z_error_far)
        
        if laneline_x_error_close.shape[0] > 0:
            x_error_close_avg = np.average(laneline_x_error_close[laneline_x_error_close > -1 + 1e-5])
        else:
            x_error_close_avg = -1
            
        if laneline_x_error_far.shape[0] > 0:
            x_error_far_avg = np.average(laneline_x_error_far[laneline_x_error_far > -1 + 1e-5])
        else:
            x_error_far_avg = -1
        if laneline_z_error_close.shape[0] > 0:
            z_error_close_avg = np.average(laneline_z_error_close[laneline_z_error_close > -1 + 1e-5])
        else:
            z_error_close_avg = -1
        if laneline_z_error_far.shape[0] > 0:
            z_error_far_avg = np.average(laneline_z_error_far[laneline_z_error_far > -1 + 1e-5])
        else:
            z_error_far_avg = -1
        
        r_lane = np.sum(self.r_list)
        p_lane = np.sum(self.p_list)
        cnt_gt = np.sum(self.cnt_gt_list)
        cnt_pred = np.sum(self.cnt_pred_list)
        Recall = r_lane / (cnt_gt + 1e-6)
        Precision = p_lane / (cnt_pred + 1e-6)
        f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
        
        dict_res = {'x_error_close':x_error_close_avg,
                    'x_error_far': x_error_far_avg,
                    'z_error_close':  z_error_close_avg,
                    'z_error_far': z_error_far_avg,
                    'recall': Recall,
                    'precision':Precision,
                    'f1_score':f1_score
                    }
        pprint(dict_res)
        return dict_res


    # def bench_all_(self,pred_lanes,gt_lanes):
    #     laneline_stats = []
    #     laneline_x_error_close = []
    #     laneline_x_error_far = []
    #     laneline_z_error_close = []
    #     laneline_z_error_far = []
    #     r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, \
    #     x_error_close, x_error_far, \
    #     z_error_close, z_error_far = self.bench(pred_lanes,
    #                                             gt_lanes)
    #     laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
    #     # consider x_error z_error only for the matched lanes
    #     # if r_lane > 0 and p_lane > 0:
    #     laneline_x_error_close.extend(x_error_close)
    #     laneline_x_error_far.extend(x_error_far)
    #     laneline_z_error_close.extend(z_error_close)
    #     laneline_z_error_far.extend(z_error_far)
    #
    #     ''' 2 '''
    #     output_stats = []
    #     laneline_stats = np.array(laneline_stats)
    #     laneline_x_error_close = np.array(laneline_x_error_close)
    #     laneline_x_error_far = np.array(laneline_x_error_far)
    #     laneline_z_error_close = np.array(laneline_z_error_close)
    #     laneline_z_error_far = np.array(laneline_z_error_far)
    #
    #     R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]) + 1e-6)  # recall = TP / (TP+FN)
    #     P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]) + 1e-6)  # precision = TP / (TP+FP)
    #     C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]) + 1e-6)  # category_accuracy
    #     F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6) #f1
    #     x_error_close_avg = np.average(laneline_x_error_close) #
    #     x_error_far_avg = np.average(laneline_x_error_far)
    #     z_error_close_avg = np.average(laneline_z_error_close)
    #     z_error_far_avg = np.average(laneline_z_error_far)
    #     # 整体的
    #     output_stats.append(F_lane) #f1
    #     output_stats.append(R_lane) #recall
    #     output_stats.append(P_lane) # percesion
    #     output_stats.append(C_lane) # categray 没用
    #     # 上边的实际上没用到
    #     output_stats.append(x_error_close_avg) # 直接求
    #     output_stats.append(x_error_far_avg) # 直
    #     output_stats.append(z_error_close_avg)
    #     output_stats.append(z_error_far_avg)
    #     # 逐帧结果
    #     output_stats.append(np.sum(laneline_stats[:, 0]))  # 8
    #     output_stats.append(np.sum(laneline_stats[:, 1]))  # 9
    #     output_stats.append(np.sum(laneline_stats[:, 2]))  # 10
    #     output_stats.append(np.sum(laneline_stats[:, 3]))  # 11
    #     output_stats.append(np.sum(laneline_stats[:, 4]))  # 12
    #     output_stats.append(np.sum(laneline_stats[:, 5]))  # 13
    #
    #     ''' 3 '''
    #     gather_output = [None for _ in range(args.world_size)] # list
    #     # all_gather all eval_stats and calculate mean
    #     dist.all_gather_object(gather_output, output_stats)
    #     r_lane = np.sum([eval_stats_sub[8] for eval_stats_sub in gather_output])
    #     p_lane = np.sum([eval_stats_sub[9] for eval_stats_sub in gather_output])
    #     c_lane = np.sum([eval_stats_sub[10] for eval_stats_sub in gather_output])
    #     cnt_gt = np.sum([eval_stats_sub[11] for eval_stats_sub in gather_output])
    #     cnt_pred = np.sum([eval_stats_sub[12] for eval_stats_sub in gather_output])
    #     match_num = np.sum([eval_stats_sub[13] for eval_stats_sub in gather_output])
    #     Recall = r_lane / (cnt_gt + 1e-6)
    #     Precision = p_lane / (cnt_pred + 1e-6)
    #     f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
    #     category_accuracy = c_lane / (match_num + 1e-6)
    #
    #     eval_stats[0] = f1_score
    #     eval_stats[1] = Recall
    #     eval_stats[2] = Precision
    #     eval_stats[3] = category_accuracy
    #     return output_stats



if __name__ == '__main__':
    pred_lanes = [np.array([[10,2],[10,10],[10,20]])]
    gt_lanes = [np.array([[10,2],[10,10],[10,20]]),np.array([[9,2],[9,10],[9,20]])]
    le = LaneEval()
    res = le.bench(pred_lanes,gt_lanes)
    print(res)