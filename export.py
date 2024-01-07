import torch
import numpy as np
from torch.nn import functional as F
from mmdet3d.apis import init_model, inference_detector
from mmcv.ops import nms, nms_rotated
from ops.voxel_module import Voxelization
from ops.iou3d_op import nms_gpu


config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'


class PointPillars(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_model(config_file, checkpoint_file, device='cpu')
        self.box_code_size = 7
        self.num_classes = 1
        self.nms_pre = 100
        self.max_num = 50
        self.score_thr = 0.1
        self.nms_thr = 0.01
        self.voxel_layer = Voxelization(voxel_size= [0.16, 0.16, 4], point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1], max_num_points=32, max_voxels=[16000, 40000])
        self.mlvl_priors = self.model.bbox_head.prior_generator.grid_anchors([torch.Size([248, 216])])
        self.mlvl_priors = [prior.reshape(-1, self.box_code_size) for prior in self.mlvl_priors]
        
    def pre_process(self, x):
        #res_voxels, res_coors, res_num_points = self.model.data_preprocessor.voxel_layer(x)
        res_voxels, res_coors, res_num_points = self.voxel_layer(x)
        return res_voxels, res_coors, res_num_points
    
    def xywhr2xyxyr(self, boxes_xywhr):
        boxes = torch.zeros_like(boxes_xywhr)
        half_w = boxes_xywhr[..., 2] / 2
        half_h = boxes_xywhr[..., 3] / 2
        boxes[..., 0] = boxes_xywhr[..., 0] - half_w
        boxes[..., 1] = boxes_xywhr[..., 1] - half_h
        boxes[..., 2] = boxes_xywhr[..., 0] + half_w
        boxes[..., 3] = boxes_xywhr[..., 1] + half_h
        boxes[..., 4] = boxes_xywhr[..., 4]
        return boxes
    
    def box3d_multiclass_nms(self, mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, mlvl_dir_scores):
        num_classes = mlvl_scores.shape[1] - 1
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        for i in range(0, num_classes):
            cls_inds = mlvl_scores[:, i] > self.score_thr
            if not cls_inds.any():
                continue
            _scores = mlvl_scores[cls_inds, i]
            _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :].cuda()
            keep = torch.zeros(_bboxes_for_nms.size(0), dtype=torch.long)
            num_out = nms_gpu(_bboxes_for_nms.cuda(), keep, self.nms_thr, _bboxes_for_nms.device.index)
            selected = keep[:num_out]
            bboxes.append(mlvl_bboxes[selected])
            scores.append(_scores[selected])
            cls_label = mlvl_bboxes.new_full((len(selected), ), i, dtype=torch.long)
            labels.append(cls_label)
            dir_scores.append(mlvl_dir_scores[selected])
        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            dir_scores = torch.cat(dir_scores, dim=0)
            if bboxes.shape[0] > self.max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:self.max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                dir_scores = dir_scores[inds]
        else:
            bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
            scores = mlvl_scores.new_zeros((0, ))
            labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
            dir_scores = mlvl_scores.new_zeros((0, ))
        return (bboxes, scores, labels, dir_scores)
    
    def decode(self, anchors, deltas):
        # xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        # xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)
        xa = anchors[:, 0].reshape(-1, 1)
        ya = anchors[:, 1].reshape(-1, 1)
        za = anchors[:, 2].reshape(-1, 1)
        wa = anchors[:, 3].reshape(-1, 1)
        la = anchors[:, 4].reshape(-1, 1)
        ha = anchors[:, 5].reshape(-1, 1)
        ra = anchors[:, 6].reshape(-1, 1)    
        xt = deltas[:, 0].reshape(-1, 1)
        yt = deltas[:, 1].reshape(-1, 1)
        zt = deltas[:, 2].reshape(-1, 1)
        wt = deltas[:, 3].reshape(-1, 1)
        lt = deltas[:, 4].reshape(-1, 1)
        ht = deltas[:, 5].reshape(-1, 1)
        rt = deltas[:, 6].reshape(-1, 1)
        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)
    
    def predict_by_feat_single(self, cls_score, bbox_pred, dir_cls_pred):
        priors = self.mlvl_priors[0]
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_scores = torch.max(dir_cls_pred, dim=-1)[1]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_score.sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)       
        max_scores, _ = scores.max(dim=1)
        _, topk_inds = max_scores.topk(self.nms_pre)    
        priors = priors[topk_inds, :].cpu()
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        dir_cls_scores = dir_cls_scores[topk_inds]
        #bboxes = self.model.bbox_head.bbox_coder.decode(priors, bbox_pred)
        bboxes = self.decode(priors, bbox_pred)
        mlvl_bboxes_bev =  torch.cat([bboxes[:, 0:2], bboxes[:, 3:5], bboxes[:, 5:6]], dim=1)
        mlvl_bboxes_for_nms = self.xywhr2xyxyr(mlvl_bboxes_bev)    
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)       
        results = self.box3d_multiclass_nms(bboxes, mlvl_bboxes_for_nms, scores, dir_cls_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:   
            dir_rot = bboxes[..., 6] + np.pi/2 - torch.floor(bboxes[..., 6] + np.pi/2 / np.pi ) * np.pi
            bboxes[..., 6] = (dir_rot - np.pi/2 + np.pi * dir_scores.to(bboxes.dtype))         
        #return mlvl_bboxes, mlvl_scores
        return bboxes, scores, labels
            
    def forward(self, res_voxels, res_coors, res_num_points):  
        voxels, coors, num_points = [], [], []
        res_coors = F.pad(res_coors, (1, 0), mode='constant', value=0)
        voxels.append(res_voxels)
        coors.append(res_coors)
        num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        num_points = torch.cat(num_points, dim=0)
        x = self.model.voxel_encoder(voxels, num_points, coors) 
        x = self.model.middle_encoder(x, coors, batch_size=1)         
        x = self.model.backbone(x)
        x = self.model.neck(x)  
        cls_scores, bbox_preds, dir_cls_preds = self.model.bbox_head(x)    
        return cls_scores[0], bbox_preds[0], dir_cls_preds[0]
        # num_levels = len(cls_scores)
        # featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # self.mlvl_priors = self.model.bbox_head.prior_generator.grid_anchors(featmap_sizes)
        # self.mlvl_priors = [prior.reshape(-1, self.box_code_size) for prior in self.mlvl_priors]
        # cls_score_list = [cls_scores[i][0].detach() for i in range(num_levels)]
        # bbox_pred_list = [bbox_preds[i][0].detach() for i in range(num_levels)]
        # dir_cls_pred_list = [dir_cls_preds[i][0].detach() for i in range(num_levels)]
        # results = self.predict_by_feat_single(cls_score_list, bbox_pred_list, dir_cls_pred_list)
        # return results
        
        
if __name__ == '__main__':        
    model = PointPillars().eval()
    # points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
    # points = torch.from_numpy(points.reshape(-1, 4))
    # res_voxels, res_coors, res_num_points = model.pre_process(points)
    res_voxels = torch.zeros(3945, 32, 4, device='cpu', dtype=torch.float32)
    res_coors = torch.zeros(3945, 3, device='cpu', dtype=torch.int32)
    res_num_points = torch.zeros(3945, device='cpu', dtype=torch.int32)
    torch.onnx.export(model, (res_voxels, res_coors, res_num_points), "pointpillars.onnx", opset_version=13)
    traced_script_module = torch.jit.trace(model, (res_voxels, res_coors, res_num_points))
    traced_script_module.save("pointpillars.pt")
