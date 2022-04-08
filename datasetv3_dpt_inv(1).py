import cv2
import matplotlib.image as mpimg
import torch
from torchvision.transforms import Compose, CenterCrop
from torchvision.transforms.functional import hflip, rotate
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.helpers import sparse_collate_tensors
from torchsparse.utils import sparse_quantize
import torch.nn as nn
import sys
import os
import random
import numpy as np
import csv
import json
from AdelaiDepth.LeReS.lib.test_utils import init_image_coor, depth_to_pcd, pcd_to_sparsetensor
from pycocotools.coco import COCO

def _read_depth(depth_path, m_factor, dataset): # max_depth param
    """returns depth tensor and mask tensor for depth, specific for each dataset"""
    
    if dataset == "DIML_outdoor":# confidence map in folder in same dir as depth folder
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth_img *= m_factor
        mask_path = os.path.join(os.path.dirname(os.path.dirname(depth_path))+"/conf", os.path.basename(depth_path)[:-9]+"conf.png")
        depth_mask = mpimg.imread(mask_path)
        gt_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        mask_tensor = ((torch.from_numpy(depth_mask).unsqueeze(0) > 0.4) & (gt_tensor >= 1e-6)) & (gt_tensor < 20.)
        return gt_tensor, mask_tensor  
    
    elif dataset == "DIODE": # confidence maps in same dir as depth
        depth_img = np.load(depth_path).astype(np.float64).squeeze(2)
        depth_img *= m_factor
        depth_mask = np.load(depth_path[:-4]+"_mask.npy")
        gt_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        mask_tensor = torch.where(((torch.from_numpy(depth_mask).unsqueeze(0) > 0.7) & (gt_tensor >= 1e-8)) & (gt_tensor < 65), 1., 0.) # 350 max depth range diode
        return gt_tensor, mask_tensor  
            
    elif dataset == "freiburg_forest": # confidence map in folder in same dir as depth folder
        raise Exception("not implemented yet")
    
    elif dataset == "lindenthal": # all values > eps are valid values
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_img *= m_factor
        gt_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        mask_tensor = torch.where((gt_tensor > 1.3) & (gt_tensor < 65.53), 1., 0.) # max depth 65.53...
        return gt_tensor, mask_tensor
    
    elif dataset == "low_viewpoint_depth": # all values > eps are valid values
        depth_img = np.array(mpimg.imread(depth_path))
        depth_img *= m_factor 
        gt_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        mask_tensor = torch.where((gt_tensor >= 1e-8) & (gt_tensor < 10.), 1., 0.) # max depth
        return gt_tensor, mask_tensor
    
    elif dataset == "TartanAir": # alle values > eps and < 5000 are valid values
        depth_img = np.load(depth_path).astype(np.float64)
        depth_img *= m_factor 
        gt_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        mask_tensor = torch.where((gt_tensor >= 1e-8) & (gt_tensor < 65), 1., 0.) # unlimited depth range, here limit set to 200, no higher ambitios, weighted loss for these will be near zero
        return gt_tensor, mask_tensor
    
    elif dataset == "UASOL": # alle values > eps and < max_depth are valid values
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth_img *= m_factor 
        gt_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        mask_tensor = torch.where((gt_tensor >= 1e-8) & (gt_tensor < 20.),1., 0.)
        return gt_tensor, mask_tensor

class WALD_PCM_Zero(Dataset):
    """Dataset Class for WALD Dataset with Zero Shot approach,
    no train/test split
    each output element consists of dpt-out as pointcloud (normed, unprojected), scale, shift, focal length
    """
    def __init__(self, csv_file_list, ds_weights, transforms_dpt, transforms_gt, train=True, inference=False):
        
        self.train = train
        self.inference = inference
        self.transforms_dpt = transforms_dpt
        self.transforms_gt = transforms_gt
        self.entries = [] 
        self.datasets = []
        self.ds_weights = {}
        
        for index_csv, csv_path in enumerate(csv_file_list):
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for i, row in enumerate(reader):
                    if i == 0:
                        self.datasets.append(row["dataset"])
                        self.ds_weights[row["dataset"]] = ds_weights[index_csv]
                    if row["dataset"] == "TartanAir":
                        if int(os.path.basename(row["dpt_path"])[:-9]) % 3 == 0 or "neighborhood" in row["dpt_path"]:
                            continue
                        
                    self.entries.append((row["dpt_path"], float(row['focal_length']),
                                        row["dataset"],row["depth_path"], float(row["m_factor"])))
                        
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # apply dpt transforms e.g. normalize, shift, invert to depth, etc.
        dpt_img = cv2.imread(entry[0], cv2.IMREAD_UNCHANGED).astype(np.float64)
        dpt_pcd = np.copy(dpt_img)
        
        # disparity to relative depth
        dpt_pcd -= dpt_pcd.min()
        dpt_pcd /= dpt_pcd.max()
        dpt_pcd = 1./(dpt_pcd*0.5+0.02)
        
        dpt_pcd_tensor = torch.from_numpy(dpt_pcd).unsqueeze(0)
        depth_tensor, mask_tensor = _read_depth(entry[3], entry[4], entry[2])
        depth_tensor = depth_tensor.float()
        mask_tensor = mask_tensor.float()
        #print(depth_tensor.dtype, mask_tensor.dtype)
        depth_tensor_masked = torch.where(mask_tensor == 1., depth_tensor, torch.tensor(0.))
        
        focal_length = entry[1]
        
        # resize to original size
        dpt_shape = tuple(dpt_pcd_tensor.shape[-2:])
        gt_shape = tuple(depth_tensor.shape[-2:])
        
        if dpt_shape != gt_shape:
            dpt_pcd_tensor =torch.nn.functional.interpolate(
                            dpt_pcd_tensor.unsqueeze(0),
                            size=gt_shape,
                            mode="bicubic",
                            align_corners=False,).squeeze(0)
            
        # adapt focal length accordingly
        
        # data augmentation
        if self.train:
            # horizontal flip
            if random.randint(1, 2) == 1:
                dpt_pcd_tensor = hflip(dpt_pcd_tensor)
                depth_tensor_masked = hflip(depth_tensor_masked)
            # rotation
            angle = np.random.uniform(-5.,5.)
            dpt_pcd_tensor = rotate(dpt_pcd_tensor, angle)
            depth_tensor_masked = rotate(depth_tensor_masked, angle)
            
                
            
            #select random centered crop and and resize to height 384
            # 1. determine if 4:3 or 16:9 crop
            if random.randint(1, 2) ==1: # 4:3
                goal_width = int(gt_shape[0] * 4 / 3)
                goal_height = gt_shape[0]
                
                # crop from 
                scale_factor = np.random.uniform(0.75, 1.) # = 1./ scale_factor
                scaled_width = int(scale_factor * goal_width)
                scaled_height =  int(scale_factor * goal_height)
                
                dpt_pcd_tensor = CenterCrop((scaled_height, scaled_width))(dpt_pcd_tensor)
                
                depth_tensor_masked = CenterCrop((scaled_height, scaled_width))(depth_tensor_masked)
                
                # resize to gt shape
                focal_length *= gt_shape[0] / scaled_height 
                dpt_pcd_tensor =torch.nn.functional.interpolate(
                                dpt_pcd_tensor.unsqueeze(0),
                                size=(goal_height, goal_width),
                                mode="bicubic",
                                align_corners=False,).squeeze()
                # resize of depth not necessary later resized to 480x640 either way
                
                # TODO debugging:
                depth_tensor_masked /= (gt_shape[0] / scaled_height)
                 
            else: # 16:9
                width_orig = dpt_pcd_tensor.shape[-1]
                goal_height =dpt_pcd_tensor.shape[-2]
                
                goal_width = width_orig
                
                if goal_width / goal_height <= 4.5 / 3: # 4:3 dpt image, needs to be cropped differently
                    goal_height = int(dpt_pcd_tensor.shape[-1] * 9 / 16)
                    
                # crop 
                scale_factor = np.random.uniform(0.75, 1.)
                scaled_width = int(scale_factor * goal_width)
                scaled_height = int(scale_factor * goal_height)
                dpt_pcd_tensor = CenterCrop((scaled_height, scaled_width))(dpt_pcd_tensor)
                
                #depth_tensor_masked = CenterCrop((int(gt_shape[1]* 9 / 16 * scale_factor), int(gt_shape[1] *scale_factor)))(depth_tensor_masked)
                depth_tensor_masked = CenterCrop((scaled_height, scaled_width))(depth_tensor_masked)
                # resize
                focal_length *=  width_orig / scaled_width  # = 1./ scale_factor
                dpt_pcd_tensor =torch.nn.functional.interpolate(dpt_pcd_tensor.unsqueeze(1),
                                                                size=(goal_height, gt_shape[1]),
                                                                mode="bicubic",
                                                                align_corners=False,).squeeze()
                # TODO debugging:
                depth_tensor_masked /= (width_orig / scaled_width)
                #print("16:9, shape", dpt_pcd_tensor.shape)
        
            # print("dpt pcd, dpt_img, depth_tensor_gt", dpt_pcd.shape, dpt_img.shape, depth_tensor.shape)

        dpt_sparse, dpt_normalized = self.transforms_dpt((dpt_pcd_tensor,focal_length)) #dpt img and focal length
        
        # apply transformation to dpt_normalized, depth_gt tensor and depth mask
        # resize them all to same res across datasets
        
    
        if self.train:
            dpt_normalized, depth_tensor_masked = self.transforms_gt((dpt_normalized, depth_tensor_masked))
        else:
            dpt_normalized = torch.nn.functional.interpolate(torch.from_numpy(dpt_pcd).unsqueeze(0).unsqueeze(0),
                                                                size=gt_shape,
                                                                mode="bicubic",
                                                                align_corners=False,).squeeze(0)
        
        if self.inference:
            return {'lidar':dpt_sparse, 'dataset': entry[2], 'dpt_normalized_tensor': dpt_normalized, 'dpt_path': entry[0]}
        if not self.train: # testing
            return{'lidar':dpt_sparse, 'dataset': entry[2],
                 'depth_tensor': depth_tensor_masked, 'dpt_normalized_tensor': dpt_normalized, 'dpt_path': entry[0]}
        # train
        return {'lidar':dpt_sparse, 'dataset': entry[2],
                 'depth_tensor': depth_tensor_masked, 'dpt_normalized_tensor': dpt_normalized}#, 'dpt_path':entry[0]} 
    
    def get_samples_weights(self, indices):
        class_count = {}
        for index in indices:
            class_count[self.entries[index][2]] = class_count[self.entries[index][2]] +1 if self.entries[index][2] in class_count else 1
        
        samples_weight = np.array([1./class_count[self.entries[index][2]] * self.ds_weights[self.entries[index][2]] for index in indices])
        return samples_weight


class LindenthalTest(Dataset):
    """Dataset Class for WALD Dataset with Zero Shot approach,
    no train/test split
    each output element consists of dpt-out as pointcloud (normed, unprojected), scale, shift, focal length
    """
    def __init__(self, lindenthal_csv, lindenthal_json_train, lindenthal_json_test, transforms_dpt, instance_wise=False):
        
        self.instance_wise = instance_wise
        self.transforms_dpt = transforms_dpt
        self.entries = [] # each entry consists of (dpt_img_path, instance_masked_depth)
        
        # row["dpt_path"], float(row['focal_length']),
        # row["dataset"],row["depth_path"], float(row["m_factor"], instace_masked_depth
        with open(lindenthal_csv, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile) 
                
            for json_file in [lindenthal_json_train, lindenthal_json_test]:
                coco_obj = COCO(json_file)
                imgIds = coco_obj.getImgIds()
                for img_id in imgIds:
                    # find corresponding image in csv files
                    img_ann = coco_obj.loadImgs([img_id])[0]
                    

                    file_name = img_ann["file_name"]
                    # check if file_name is an interpolated image and skip it, if that is the case
                    if (not (int(file_name[-10:-4]) >= 31) or (int(file_name[-10:-4]) - 31) % 10 != 0):
                        #print("skipped frame ", file_name)
                        continue
                       
                    # find corresponding row in csv file
                    csvfile.seek(0)
                    for row in csv_reader:
                        if row["RGB_path"][27:] == file_name:
                            break
                    else: # file_name is not in csv file
                        print("file name not in csv file: ", file_name,  row["RGB_path"][27:])
                        continue
                        
                    
                    # create mask for current img
                    anns_ids = coco_obj.getAnnIds(imgIds=[img_id], iscrowd=None)
                    anns = coco_obj.loadAnns(anns_ids)
                    instance_masked_depth = np.zeros((img_ann['height'],img_ann['width']))
                    if self.instance_wise:
                        for i, ann in enumerate(anns):
                            if 'track_id' not in ann['attributes'].keys():
                                continue
                            instance_masked_depth[coco_obj.annToMask(ann) == 1] = i+1
                    else:
                        for ann in anns:
                            instance_masked_depth[coco_obj.annToMask(ann) == 1] = 1
                    
                    # append to entries
                    self.entries.append((row["dpt_path"], float(row['focal_length']), row["dataset"], 
                                        row["depth_path"], float(row["m_factor"]), instance_masked_depth))                               
                                            
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # apply dpt transforms e.g. normalize, shift, invert to depth, etc.
        dpt_img = cv2.imread(entry[0], cv2.IMREAD_UNCHANGED).astype(np.float64)
        dpt_pcd = np.copy(dpt_img)
        
        # disparity to relative depth
        dpt_pcd -= dpt_pcd.min()
        dpt_pcd /= dpt_pcd.max()
        dpt_pcd = 1./(dpt_pcd*0.5+0.02)
        
        dpt_pcd_tensor = torch.from_numpy(dpt_pcd).unsqueeze(0)
        
        depth_tensor, mask_tensor = _read_depth(entry[3], entry[4], entry[2])
        depth_tensor = depth_tensor.float()
        mask_tensor = mask_tensor.float()
        instance_mask = torch.from_numpy(entry[5]).unsqueeze(0)
        
        #print(depth_tensor.dtype, mask_tensor.dtype)
        if self.instance_wise:
            depth_tensor_masked = torch.where(mask_tensor == 1., depth_tensor, torch.tensor(0.))
        else:
            depth_tensor_masked = torch.where((mask_tensor == 1.) & (instance_mask == 1.), depth_tensor, torch.tensor(0.))
            
            
        focal_length = entry[1]
        
        # resize to orignal size
        dpt_shape = tuple(dpt_pcd_tensor.shape[-2:])
        gt_shape = tuple(depth_tensor.shape[-2:])
        
        if dpt_shape != gt_shape:
            dpt_pcd_tensor =torch.nn.functional.interpolate(
                            dpt_pcd_tensor.unsqueeze(0),
                            size=gt_shape,
                            mode="bicubic",
                            align_corners=False,).squeeze(0)
        # adapt focal length accordingly
        #focal_length *= dpt_pcd_tensor.shape[-2] / gt_shape[-2]
        
        
            
        dpt_sparse, dpt_normalized = self.transforms_dpt((dpt_pcd_tensor, focal_length)) #dpt img and focal length
        
        # resize dpt_normalized to original resolution for metrics
        dpt_normalized =torch.nn.functional.interpolate(
                            dpt_normalized,
                            size=gt_shape,
                            mode="bicubic",
                            align_corners=False,).squeeze(0)
                
        if self.instance_wise:
            return {'lidar':dpt_sparse, 'dataset': 'lindenthal',
                 'depth_tensor': depth_tensor_masked, 'dpt_normalized_tensor': dpt_normalized, 'instance_mask': instance_mask, 'dpt_path': entry[0]} 
        else:
            return {'lidar':dpt_sparse, 'dataset': 'lindenthal',
                     'depth_tensor': depth_tensor_masked, 'dpt_normalized_tensor': dpt_normalized, 'dpt_path': entry[0]} 


def get_transforms_gt(width=640, height=480):
    def _resize(double_tuple):
        dpt_norm, depth_tensor_masked = double_tuple
        #print("dpt, depth, mask", dpt_norm.shape, depth_tensor.shape, mask_tensor.shape)
        dpt_norm_res = nn.functional.interpolate(dpt_norm.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=False)
        depth_tensor_masked_res = nn.functional.interpolate(depth_tensor_masked.unsqueeze(0), size=(height, width), mode="nearest")
        
        return dpt_norm_res.squeeze(0), depth_tensor_masked_res.squeeze(0)
    return Compose([_resize])


def get_transforms_dpt(voxel_size=0.01, num_points=100000):
    
    def dpt_2_pcd(dpt_tupel):
        # reconstruct PCD from depth
        dpt_normalized = dpt_tupel[0] # tensor
        focal_length = dpt_tupel[1]
        cam_u0 = dpt_normalized.shape[-1] / 2.0
        cam_v0 = dpt_normalized.shape[-2] / 2.0
        u_u0, v_v0 = init_image_coor(dpt_normalized.shape[-2], dpt_normalized.shape[-1], u0=cam_u0, v0=cam_v0)
        pcd_dpt, mask_valid = depth_to_pcd(dpt_normalized.squeeze(0).numpy(), u_u0, v_v0, f=focal_length, invalid_value=0)
        #print('pcd', pcd_dpt)
        # input for the voxelnet
        lidar = pcd_to_sparsetensor_custom(pcd_dpt, mask_valid, voxel_size=voxel_size, num_points=num_points) # 0.01 original voxel size
        return lidar, dpt_normalized.unsqueeze(0)
    
    return Compose([dpt_2_pcd])

def sparse_collate_fn_train(batch):
    if isinstance(batch[0], dict):
        batch_size = batch.__len__()
        ans_dict = {}
        for key in batch[0].keys():
            
            if isinstance(batch[0][key], SparseTensor):
                ans_dict[key] = sparse_collate_tensors(
                    [sample[key] for sample in batch])
            elif isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch],
                    axis=0)
            
                
            elif isinstance(batch[0][key], torch.Tensor): 
                ans_dict[key] = torch.stack([sample[key] for sample in batch],
                                            axis=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = sparse_collate_fn_custom(
                    [sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
    else:
        batch_size = batch.__len__()
        ans_dict = tuple()
        for i in range(len(batch[0])):
            key = batch[0][i]
            if isinstance(key, SparseTensor):
                ans_dict += sparse_collate_tensors(
                    [sample[i] for sample in batch]),
            elif isinstance(key, np.ndarray):
                ans_dict += torch.stack(
                    [torch.from_numpy(sample[i]) for sample in batch], axis=0),
            elif isinstance(key, torch.Tensor):
                ans_dict += torch.stack([sample[i] for sample in batch],
                                        axis=0),
            elif isinstance(key, dict):
                ans_dict += sparse_collate_fn_custom([sample[i] for sample in batch]),
            else:
                ans_dict += [sample[i] for sample in batch],
        return ans_dict


def sparse_collate_fn_test(batch):
    if isinstance(batch[0], dict):
        batch_size = batch.__len__()
        ans_dict = {}
        for key in batch[0].keys():
            
            if isinstance(batch[0][key], SparseTensor):
                ans_dict[key] = sparse_collate_tensors(
                    [sample[key] for sample in batch])
            elif isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch],
                    axis=0)
            elif key == 'depth_tensor' or key == 'mask_tensor' or key == 'dpt_normalized_tensor':
                
                ans_dict[key] = [sample[key] for sample in batch]
                
            elif isinstance(batch[0][key], torch.Tensor): 
                ans_dict[key] = torch.stack([sample[key] for sample in batch],
                                            axis=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = sparse_collate_fn_custom(
                    [sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        return ans_dict
    else:
        batch_size = batch.__len__()
        ans_dict = tuple()
        for i in range(len(batch[0])):
            key = batch[0][i]
            if isinstance(key, SparseTensor):
                ans_dict += sparse_collate_tensors(
                    [sample[i] for sample in batch]),
            elif isinstance(key, np.ndarray):
                ans_dict += torch.stack(
                    [torch.from_numpy(sample[i]) for sample in batch], axis=0),
            elif isinstance(key, torch.Tensor):
                ans_dict += torch.stack([sample[i] for sample in batch],
                                        axis=0),
            elif isinstance(key, dict):
                ans_dict += sparse_collate_fn_custom([sample[i] for sample in batch]),
            else:
                ans_dict += [sample[i] for sample in batch],
        return ans_dict


def pcd_to_sparsetensor_custom(pcd, mask_valid, voxel_size=0.01, num_points=100000):
    pcd_valid = pcd[mask_valid]
    block_ = pcd_valid
    block = np.zeros_like(block_)
    block[:, :3] = block_[:, :3]

    pc_ = np.round(block_[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = block

    # transfer point cloud to voxels
    inds = sparse_quantize(pc_,
                           feat_,
                           return_index=True,
                           return_invs=False)
    if len(inds) > num_points:
        inds = np.random.choice(inds, num_points, replace=False)

    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(feat, pc)
    # feed_dict = [{'lidar': lidar}]
    # inputs = sparse_collate_fn(feed_dict)
    return lidar

    
