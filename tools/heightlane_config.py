import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from loader.bev_road.openlane_data import OpenLane_dataset_with_offset,OpenLane_dataset_with_offset_val
from models.model.heightlane import HeightLane

ROOT_DIR = "/home/work/chase_data/openlane/"

# Data splits
train = {
    "gt": f"{ROOT_DIR}training",
    "images": f"{ROOT_DIR}images/training",
    "maps": f"{ROOT_DIR}heightmap_training"
}

val = {
    "gt": f"{ROOT_DIR}validation",
    "images": f"{ROOT_DIR}images/validation",
    "maps": f"{ROOT_DIR}heightmap_validation"
}

model_save_path = "./heightlane"

input_shape = (600,800)
output_2d_shape = (144,256)

''' BEV range '''
x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5 # grid size
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel),int((y_range[1] - y_range[0]) / meter_per_pixel))

loader_args = dict(
    batch_size=8,
    num_workers=12
)

val_loader_args = dict(
    batch_size=24,
    num_workers=4
)

''' model '''
def model():
    return HeightLane(bev_shape=bev_shape, image_shape = input_shape, output_2d_shape=output_2d_shape,train=True)

''' optimizer '''
epochs = 50
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR



def train_dataset():
    train_trans = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.MotionBlur(p=0.2),
        A.RandomBrightnessContrast(),
        A.ColorJitter(p=0.1),
        A.Normalize(),
        ToTensorV2()
    ])
    
    train_data = OpenLane_dataset_with_offset(
        train["images"], train["gt"], train["maps"],
        x_range, y_range, meter_per_pixel,
        train_trans, input_shape, output_2d_shape
    )
    
    return train_data

def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2()
    ])
    
    val_data = OpenLane_dataset_with_offset_val(
        val["images"], val["gt"], val["maps"],
        trans_image
    )
    
    return val_data


