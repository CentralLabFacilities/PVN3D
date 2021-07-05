#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from pvn3d.common import Config
from pvn3d.lib import PVN3D
from pvn3d.datasets.ycb.ycb_image import YCB_Image
from pvn3d.lib.utils.sync_batchnorm import convert_model
from pvn3d.lib.utils.pvn3d_eval_utils import cal_frame_poses, cal_frame_poses_lm
from pvn3d.lib.utils.basic_utils import Basic_Utils
try:
    from cv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-image", type=str, default=None, help="Image to eval"
)
args = parser.parse_args()

config = Config(dataset_name='ycb')
bs_utils = Basic_Utils(config)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        try:
            checkpoint = torch.load(filename)
        except:
            checkpoint = pkl.load(open(filename, "rb"))
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def cal_view_pred_pose(model, data, epoch=0, saveImages=True):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = [item.to("cuda", non_blocking=True) for item in data]
        rgb, pcld, cld_rgb_nrm, choose = cu_dt #, cls_ids
        pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
            cld_rgb_nrm, rgb, choose
        )
        _, classes_rgbd = torch.max(pred_rgbd_seg, -1)

        pred_cls_ids, pred_pose_lst = cal_frame_poses(
            pcld[0], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
            config.n_objects, True
        )
        print("ids", pred_cls_ids)
        print("pose", pred_pose_lst)
        if saveImages:
            np_rgb = rgb.cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
            np_rgb = np_rgb[:, :, ::-1].copy()
            for idx, cls_id in enumerate(pred_cls_ids):
                pose = pred_pose_lst[idx]
                obj_id = int(cls_id)
                mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type='ycb').copy()
                mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
                K = config.intrinsic_matrix["ycb_K1"]
                mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
                color = bs_utils.get_label_color(obj_id, n_obj=22, mode=1)
                np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
            vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
            ensure_fd(vis_dir)
            f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
            cv2.imwrite(f_pth, np_rgb)
        if epoch == 0 and saveImages:
            print("\n\nResults saved in {}".format(vis_dir))


def main():
    test_ds = YCB_Image(args.image)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=1
    )

    model = PVN3D(
        num_classes=config.n_objects, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=config.n_sample_points
    ).cuda()
    model = convert_model(model)
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )
    model = nn.DataParallel(model)

    for i, data in tqdm.tqdm(enumerate(test_loader), leave=False, desc="val"):
        cal_view_pred_pose(model, data, epoch=i)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
