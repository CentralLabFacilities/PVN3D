#!/usr/bin/env python3
import os
import cv2
import pcl
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pvn3d.common import Config
import pickle as pkl
from pvn3d.lib.utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
from cv2 import imshow, waitKey


config = Config(dataset_name='ycb')
bs_utils = Basic_Utils(config)


class YCB_Image():

    def __init__(self, path):
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.diameters = {}
        self.cls_lst = bs_utils.read_lines(config.ycb_cls_lst_p)
        self.obj_dict = {}
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
        self.path = path #'/mnt/c/Users/Jan/source/repos/ethnhe/PVN3D/pvn3d/datasets/ycb/dataset_config/test_data_list_kurz.txt'
        self.sym_cls_ids = [13, 16, 19, 20, 21]
        self.cam_scale = 10000 #factor_depth

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def get_normal(self, cld):
        cloud = pcl.PointCloud()
        cld = cld.astype(np.float32)
        cloud.from_array(cld)
        ne = cloud.make_NormalEstimation()
        kdtree = cloud.make_kdtree()
        ne.set_SearchMethod(kdtree)
        ne.set_KSearch(50)
        n = ne.compute()
        n = n.to_array()
        return n

    def get_item(self):
        try:
            dpt = np.array(Image.open(self.path + '-depth.png'))
            # meta = scio.loadmat(os.path.join(self.path,'-meta.mat'))
            K = config.intrinsic_matrix['ycb_K1']
            rgb = np.array(Image.open(self.path+'-color.png'))[:, :, :3]

            dpt = bs_utils.fill_missing(dpt, self.cam_scale, 1)

            rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
            cld, choose = bs_utils.dpt_2_cld(dpt, self.cam_scale, K)
            normal = self.get_normal(cld)[:, :3]
            normal[np.isnan(normal)] = 0.0

            rgb_lst = []
            for ic in range(rgb.shape[0]):
                rgb_lst.append(
                    rgb[ic].flatten()[choose].astype(np.float32)
                )
            rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()

            choose = np.array([choose])
            choose_2 = np.array([i for i in range(len(choose[0, :]))])

            if len(choose_2) < 400:
                return None
            if len(choose_2) > config.n_sample_points:
                c_mask = np.zeros(len(choose_2), dtype=int)
                c_mask[:config.n_sample_points] = 1
                np.random.shuffle(c_mask)
                choose_2 = choose_2[c_mask.nonzero()]
            else:
                choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')

            cld_rgb_nrm = np.concatenate((cld, rgb_pt, normal), axis=1)
            cld = cld[choose_2, :]
            cld_rgb_nrm = cld_rgb_nrm[choose_2, :]
            choose = choose[:, choose_2]

            # cls_ids = np.zeros((config.n_objects, 1))
            # cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
            # for i, cls_id in enumerate(cls_id_lst):
            #     cls_ids[i, :] = np.array([cls_id])

            # rgb, pcld, cld_rgb_nrm, choose, cls_ids
            return  torch.from_numpy(rgb.astype(np.float32)), \
                    torch.from_numpy(cld.astype(np.float32)), \
                    torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
                    torch.LongTensor(choose.astype(np.int32)), #\
                    # torch.LongTensor(cls_ids.astype(np.int32)),
        except:
            return None


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.get_item()

# vim: ts=4 sw=4 sts=4 expandtab
