import torch
import torchvision as tv
from torch import nn
from functools import reduce
import torch.nn.functional as F
import pdb
from scipy.interpolate import splev
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .deformable import _get_activation_fn, IdentityMSDeformAttn, DropoutMSDeformAttn, PositionEmbeddingLearned

class FFN(nn.Module):

    def __init__(self,
                d_model=256,
                dim_ff=1024,
                activation='relu',
                ffn_dropout=0.,
                add_identity=True):
        super().__init__()

        self.d_model = d_model
        self.feedforward_channels = dim_ff

        self.linear1 = nn.Linear(d_model, dim_ff)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(ffn_dropout)

        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.add_identity = add_identity
        self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight.data)
        constant_(self.linear1.bias.data, 0.)
        xavier_uniform_(self.linear2.weight.data)
        constant_(self.linear2.bias.data, 0.)


    def forward(self, x, identity=None):
        inter = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        out = self.dropout2(inter)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class EncoderLayer(nn.Module):

    '''
        one layer in Encoder,
        self-attn -> norm -> cross-attn -> norm -> ffn -> norm

        INIT:   d_model: this is C in ms uv feat map & BEV feat map
                dim_ff: num channels in feed forward net (FFN)
                activation, ffn_dropout: used in FFN
                num_levels: num layers of fpn out
                num_points, num_heads: used in deform attn
    '''
    def __init__(self,
                 d_model=None,
                 dim_ff=None,
                 activation="relu",
                 ffn_dropout=0.0,
                 num_levels=4,
                 num_points=8,
                 num_heads=8):
        super().__init__()
        self.fp16_enabled = False

        self.self_attn = IdentityMSDeformAttn(d_model=d_model, n_levels=1)  # q=v,
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = DropoutMSDeformAttn(d_model=d_model, 
                                        n_levels=num_levels, 
                                        n_points=num_points, 
                                        n_heads=num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model=d_model, dim_ff=dim_ff, activation=activation,
                                     ffn_dropout=ffn_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    '''
        INPUT:  query: (B, bev_h*bev_w, C), this is BEV feat map
                value: (B, \sum_{l=0}^{L-1} H_l \cdot W_l, C), this is ms uv feat map from FPN, C is fixed for all scale
                bev_pos: BEV feat map pos embed (B, bev_h*bev_w, C)
                ref_2d: ref pnts used in self-attn, for query (B, bev_h*bev_w, 1, 2)
                ref_3d: ref pnts used in cross-attn, for ms uv feat map from FPN, this is IMPORTANT for uv-bev transform
                        (B, bev_h*bev_w, 4, 2)
                bev_h: height of bev feat map
                bev_w: widght of bev feat map
                spatial_shapes: input spatial shapes for cross-attn, this is used to split ms uv feat map
                level_start_index: input level start index for cross-attn, this is used to split ms uv feat map
                
            self-attn:
                input: q=v=query, ref_pnts = ref_2d (universal sampling over query space), 1-lvl
                output: query for cross-attn
            
            cross-attn:
                input: q=query, v=value=ms_uv_feat_map, ref_pnts = ref_3d (this is projection from bev loc to uv loc, 
                                                                            so that attention of each bev loc 
                                                                            can focus on relative uv loc)
                output: bev feat map
    '''
    def forward(self,
                query=None,
                value=None,
                bev_pos=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                spatial_shapes=None,
                level_start_index=None):
        
        # self attention
        identity = query

        temp_key = temp_value = query
        query = self.self_attn( query + bev_pos,
                                reference_points=ref_2d,
                                input_flatten=temp_value,
                                input_spatial_shapes=torch.tensor(
                                    [[bev_h, bev_w]], device=query.device),
                                input_level_start_index=torch.tensor(
                                    [0], device=query.device),
                                identity=identity)
        identity = query

        # norm 1
        query = self.norm1(query)

        # cross attention
        query = self.cross_attn(query,
                                reference_points=ref_3d,
                                input_flatten=value,
                                input_spatial_shapes=spatial_shapes,
                                input_level_start_index=level_start_index)
        query = query + identity

        # norm 2
        query = self.norm2(query)

        # ffn
        query = self.ffn(query)

        # norm 3
        query = self.norm3(query)

        return query
    
def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)  
            nn.init.constant_(m.bias, 0)
    return mod

def mul(a,b):
    return a*b

class JustHeight(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(JustHeight, self).__init__()
        
        self.bev_up_height = nn.Sequential(
            nn.Conv2d(input_channel, 1024, 1, 1, 1, bias=False),
            # nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                ),
                downsample=nn.Conv2d(1024, 256, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(256, 128, 1),
            ),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )
        
        naive_init_module(self.bev_up_height)

    def forward(self, bev_x):
        bev_height = self.bev_up_height(bev_x)
        return bev_height

class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )
        
        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )
                
        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )
        
        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        # naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        # pred, emb, offset_y, z
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat) #, self.m_z(feat)

class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)
class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()
        
        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)

    
class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),
            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )
        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)

class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
        
class HeightLane(nn.Module):  
    # Multi heightmap initialize
    def create_slope_heightmap(self, angle, bev_shape=(200,48)):
        # Convert angle from degrees to radians
        rad = torch.tensor(angle * (torch.pi / 180))
        # Calculate the height increment per pixel
        height_increment = torch.tan(rad)
        # Create a tensor with the height range and multiply by height increment
        height_tensor = torch.arange(bev_shape[0], dtype=torch.float32) * height_increment
        # Invert the height tensor to have the maximum height at the first row
        height_tensor = height_tensor.flip(0)
        # Expand the height_tensor to match the width dimension
        heightmap = height_tensor.unsqueeze(1).expand(-1, bev_shape[1])
        return heightmap
    
    def __init__(self, bev_shape, image_shape, output_2d_shape, train=True):
        super(HeightLane, self).__init__()
        self.bb = nn.Sequential(*list(tv.models.resnet50(pretrained=True).children())[:-3])
        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)
                ),
                downsample=nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            )
        )
        
        self.maxtrix_cameraw2camera = torch.tensor(([[-0., -1., -0., -0.],
                                    [-0., -0., -1., -0.],
                                    [ 1.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  1.]]), dtype=torch.float32)
        
        
        x = torch.linspace(0, bev_shape[0] - 1, bev_shape[0]) #longitudinal
        y = torch.linspace(0, bev_shape[1] - 1, bev_shape[1]) #lateral
        self.xv_, self.yv_ = torch.meshgrid(x, y)
        
        self.height_anchor_0 =  torch.zeros((200,48), dtype=torch.float32)
        self.height_anchor_pos = self.create_slope_heightmap(5.0)
        self.height_anchor_neg = self.create_slope_heightmap(-5.0)
        
        self.height_extractor = JustHeight(bev_shape, input_channel=3072)
        # self.height_neck = HeightNeck(bev_shape, input_channel=512)
        # self.height_head = HeightHead(bev_shape, input_channel=512)
        
        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=2048)
        self.is_train = train
        if self.is_train: 
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=1024)
            
        self.num_att = 2
        self.num_proj = 2 # number of front view feature 
        self.query_embeds = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.el = nn.ModuleList()
        self.ref_2d = []
        self.input_spatial_shapes = []
        self.input_level_start_index = []
        
        ## 800 600 때 uv_h, uv_w 38, 50 // 19, 25
        
        for i in range(self.num_proj):
            bev_h, bev_w = bev_shape
            uv_feat_c = 1024
            if i ==0:
                uv_h, uv_w = 38, 50 
            if i ==1:
                uv_h, uv_w = 19, 25
                
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, bev_h - 0.5, bev_h), torch.linspace(0.5, bev_w - 0.5, bev_w)
            )
            ref_y = ref_y.reshape(-1)[None] / bev_h  # ?
            ref_x = ref_x.reshape(-1)[None] / bev_w  # ?
            ref_point = torch.stack((ref_x, ref_y), -1)
            ref_point = ref_point.repeat(1, 1, 1).unsqueeze(2)
            self.ref_2d.append(ref_point)
            bev_feat_len = bev_h * bev_w
            query_embed = nn.Embedding(bev_feat_len, uv_feat_c)
            self.query_embeds.append(query_embed)
            position_embed = PositionEmbeddingLearned(bev_h, bev_w, num_pos_feats=uv_feat_c//2)
            self.pe.append(position_embed)

            spatial_shape = torch.as_tensor([(uv_h, uv_w)], dtype=torch.long)
            self.input_spatial_shapes.append(spatial_shape)

            level_start_index = torch.as_tensor([0.0,], dtype=torch.long)
            self.input_level_start_index.append(level_start_index)

            for j in range(self.num_att):
                encoder_layers = EncoderLayer(d_model=uv_feat_c, dim_ff=uv_feat_c, num_levels=1, 
                                              num_points=4, num_heads=4)
                self.el.append(encoder_layers)
        
    def height4featuremap(self, heightmaps, intrinsics, extrinsics, featuremap, input_2d = (800,600), anchor=False):
        device = intrinsics.device
        batch_size, c1, h1, w1 = featuremap.shape
        if anchor:
            heightmaps = heightmaps.unsqueeze(0).expand(batch_size, -1, -1)
            batch_size, bev_height, bev_width = heightmaps.shape
            z_grid = heightmaps.to(device) # Remove the channel dimension
        else:
            batch_size, _, bev_height, bev_width = heightmaps.shape
            z_grid = heightmaps.squeeze(1)  # Remove the channel dimension
        
        #extrinsics = torch.inverse(extrinsics.cpu()).to(intrinsics.device)
        extrinsics = self.maxtrix_cameraw2camera.to(device) @ extrinsics.to(device)
        
        x_grid = self.xv_.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        y_grid = self.yv_.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        xv = 103 -((x_grid +0.5) / 2) 
        yv = (y_grid / 2) - 12 
        
        ones = torch.ones_like(z_grid)
        bev_coords = torch.stack((yv, xv, z_grid, ones), dim=-1).reshape(batch_size, bev_height * bev_width, 4)
        bev_coords = bev_coords.permute(0, 2, 1)  # Shape: (batch_size, 4, bev_height * bev_width)
        
        bev_points_camera = torch.bmm(extrinsics, bev_coords) # batch x 4 x 9600
        
        intrinsic1 = intrinsics.clone()
        intrinsic1[:,0,:] = intrinsic1[:,0,:] * w1 / input_2d[0]
        intrinsic1[:,1,:] = intrinsic1[:,1,:] * h1 / input_2d[1]

        bev_points_image = torch.bmm(intrinsic1, bev_points_camera[:,:3,:]).permute(0,2,1) # batch x 9600 x 3
        bev_points_image = bev_points_image[..., :2] / bev_points_image[..., 2:]
        if anchor:
            bev_points_image[..., 0] = (bev_points_image[..., 0] / (w1 - 1)) * 2 - 1
            bev_points_image[..., 1] = (bev_points_image[..., 1] / (h1 - 1)) * 2 - 1

            bev_points_image = bev_points_image.reshape(batch_size, bev_height, bev_width, 2)
            
            bev_points_image = bev_points_image.float()
            image_feature_sampled = F.grid_sample(featuremap, bev_points_image, align_corners=True)
            return image_feature_sampled
        else:
            bev_points_image[..., 0] = (bev_points_image[..., 0] / (w1 - 1)) 
            bev_points_image[..., 1] = (bev_points_image[..., 1] / (h1 - 1)) 
            bev_points_image = bev_points_image.float()
            return bev_points_image
        
    
    def forward(self, img, intrinsic, extrinsic, prev=False):
        # pdb.set_trace()
        img_s32 = self.bb(img)  # b x 1024 x 36 x 64
        bs, c, h1, w1 = img_s32.shape
        img_s64 = self.down(img_s32) # 
        bs, c, h2, w2 = img_s64.shape
        fv_features = [img_s32, img_s64]

        bev_s32_anchor0 = self.height4featuremap(self.height_anchor_0, intrinsic, extrinsic, img_s32, anchor=True)
        bev_s32_anchor_pos = self.height4featuremap(self.height_anchor_pos, intrinsic, extrinsic, img_s32, anchor=True)
        bev_s32_anchor_neg = self.height4featuremap(self.height_anchor_neg, intrinsic, extrinsic, img_s32, anchor=True)

        bev_anchor_feature = torch.cat([bev_s32_anchor_pos,bev_s32_anchor0, bev_s32_anchor_neg], dim=1)
        heightmap = self.height_extractor(bev_anchor_feature)   
        if prev:
            return heightmap     
        projs = []
        for i in range(self.num_proj):  
            src = fv_features[i].flatten(2).permute(0, 2, 1) # 8 x (h*w) x 1024
            ref_pnts = self.height4featuremap(heightmap, intrinsic, extrinsic, fv_features[i], input_2d = (800,600)).unsqueeze(-2)
            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)
            bev_pos = self.pe[i](heightmap).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
            
            ref_2d = self.ref_2d[i].repeat(bs, 1, 1, 1).to(intrinsic.device)
            input_spatial_shapes = self.input_spatial_shapes[i].to(intrinsic.device)
            input_level_start_index = self.input_level_start_index[i].to(intrinsic.device)
            for j in range(self.num_att):
                query_embed = self.el[i*self.num_att+j](query=query_embed, value=src, bev_pos=bev_pos, 
                                                                ref_2d = ref_2d, ref_3d=ref_pnts,
                                                                bev_h=200, bev_w=48, 
                                                                spatial_shapes=input_spatial_shapes,
                                                                level_start_index=input_level_start_index)
            query_embed = query_embed.permute(0, 2, 1).view(bs, c, 200, 48).contiguous()
            projs.append(query_embed)

        transformed_feature = torch.cat(projs, dim=1)
        # print(transformed_feature.shape, heightmap.shape, img_s32.shape)
        if self.is_train:
            return self.lane_head(transformed_feature) , heightmap, self.lane_head_2d(img_s32)
        else:
            return self.lane_head(transformed_feature) , heightmap
