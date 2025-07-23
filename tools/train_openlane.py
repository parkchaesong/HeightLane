import sys
sys.path.append('/home/work/chase/heightlane')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
import pdb
import torch.nn.functional as F
from datetime import timedelta
import wandb
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

class CustomSmoothL1Loss(nn.Module):
    def __init__(self):
        super(CustomSmoothL1Loss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, input, target, mask):
        # Smooth L1 Loss 계산
        loss = self.l1_loss(input, target)
        
        # 마스크 적용
        masked_loss = loss * mask
        
        loss_sum = masked_loss.sum()
        num_valid = mask.sum()
        
        return loss_sum / num_valid
    
class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.emb_loss = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # self.l1_loss = nn.SmoothL1Loss()
        self.l1_loss = CustomSmoothL1Loss()

    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, gt_height = None, intrinsic= None, road2cam = None, extrinsic= None,
                cam_w_ext=None, vehicle_pose=None, train=True):
        (pred, emb, offset_y), heightmap, (pred_2d, emb_2d) = self.model(inputs, intrinsic, road2cam)
        if train: 
            ## 3d
            gt_heightmap = gt_height[:,0,:,:].unsqueeze(1)
            gt_heightmask = gt_height[:,1,:,:].unsqueeze(1)
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
            loss_emb = self.emb_loss(emb, gt_instance)
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
            loss_height = self.l1_loss(heightmap, gt_heightmap, gt_heightmask)
            
            loss_total = 5 * loss_seg + loss_emb
            loss_total = loss_total.unsqueeze(0)
            loss_offset = 60 * loss_offset.unsqueeze(0)
            loss_height = 10 * loss_height.unsqueeze(0) 

            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
            loss_emb_2d = self.emb_loss(emb_2d, image_gt_instance)
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
            loss_total_2d = loss_total_2d.unsqueeze(0)

            return pred, loss_total, loss_offset, loss_total_2d , loss_height
        else:
            return pred, heightmap, pred_2d

def train_epoch(rank, model, dataset, optimizer, configs, epoch):
    # Last iter as mean loss of whole epoch
    model.train()
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance, intrinsic, extrinsic, road2cam, heightmap \
        , cam_w_extrinsics, vehicle_pose) in enumerate(dataset):
        
        input_data = input_data.to(rank, non_blocking=True)
        image_intrinsic = intrinsic.to(rank, non_blocking=True)
        
        image_extrinsic = extrinsic.to(rank, non_blocking=True)
        image_road2cam = road2cam.to(rank, non_blocking=True)
        gt_heightmap = heightmap.to(rank, non_blocking=True)
        gt_seg_data = gt_seg_data.to(rank, non_blocking=True)
        gt_emb_data = gt_emb_data.to(rank, non_blocking=True)
        offset_y_data = offset_y_data.to(rank, non_blocking=True)
        z_data = z_data.to(rank, non_blocking=True)
        image_gt_segment = image_gt_segment.to(rank, non_blocking=True)
        image_gt_instance = image_gt_instance.to(rank, non_blocking=True)
        
        ''' for consistency loss'''
        
        cam_w_extrinsics = cam_w_extrinsics.to(rank, non_blocking=True)
        vehicle_pose = vehicle_pose.to(rank, non_blocking=True)

        # image_gt_heatmap = image_gt_heatmap.cuda()
        # tim = time.time()
        prediction, loss_total_bev, loss_offset, loss_total_2d, loss_height= model(input_data, 
                                                                                   gt_seg_data,
                                                                                gt_emb_data,
                                                                                offset_y_data, 
                                                                                z_data,
                                                                                image_gt_segment,
                                                                                image_gt_instance,
                                                                                gt_heightmap,
                                                                                intrinsic=image_intrinsic,
                                                                                extrinsic = image_extrinsic,
                                                                                road2cam = image_road2cam,
                                                                                cam_w_ext=cam_w_extrinsics,
                                                                                vehicle_pose=vehicle_pose,
                                                                                train=True)
        
        loss_back_bev = loss_total_bev.mean()
        loss_back_2d = loss_total_2d.mean()
        loss_offset = loss_offset.mean()
        loss_height = loss_height.mean()
        loss_back_total = loss_back_bev + loss_offset + loss_back_2d + loss_height 

        ''' caclute loss '''

        optimizer.zero_grad()
        loss_back_total.backward()
        optimizer.step()
        if rank == 0:

            wandb.log({"loss_back_bev": loss_back_bev.item(), "loss_offset": loss_offset.item(), 
                "loss_height": loss_height.item(), "loss_back_total": loss_back_total.item(), "loss_back_2d": loss_back_2d.item()
             })
        if rank == 0 and idx % 50 == 0:
            print(idx, loss_back_bev.item(), '*' * 10)
            
        if idx % 300 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            loss_iter = {"BEV Loss": loss_back_bev.item(), 'offset loss': loss_offset.item(), 'height loss': loss_height.item()
                            ,"F1_BEV_seg": f1_bev_seg}
            if rank ==0:
                wandb.log({"F1-bev-seg": f1_bev_seg})
        torch.distributed.barrier()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=5))
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def worker_function(rank, world_size, config_file, checkpoint_path=None):
    setup(rank, world_size)
    print('use gpu ids is ' + ','.join([str(i) for i in range(world_size)]))
    configs = load_config_module(config_file)
    if rank == 0:
        wandb.init(project="heightlane", group='distributed_training')
    
    ''' models and optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    if torch.cuda.is_available():
        model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)
    
    if Dataset is None:
        Dataset = configs.training_dataset
    
    train_sampler = DistributedSampler(Dataset(), num_replicas=world_size, rank=rank)    
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True, sampler=train_sampler, shuffle=False,  persistent_workers=True)
    torch.cuda.empty_cache()
    for epoch in range(configs.epochs):
        print_epoch = epoch 
        dist.barrier()  # Synchronize before starting the epoch
        train_sampler.set_epoch(epoch)
        train_epoch(rank, model, train_loader, optimizer, configs, epoch)
        scheduler.step()
        if rank == 0:  # Save model only from process 0 to avoid overwriting
            save_model_dp(model, optimizer, configs.model_save_path, 'ep%03d.pth' % print_epoch)
            save_model_dp(model, None, configs.model_save_path, 'latest.pth')
    cleanup()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    mp.spawn(worker_function,
             args=(1, '/home/work/chase/heightlane/tools/heightlane_config.py'),
             nprocs=1,
             join=True)
