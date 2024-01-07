#include <iostream>
#include <vector>
#include <fstream>
#include <torch/script.h>
#include "voxelization.h"
#include "iou3d.h"

#define _USE_MATH_DEFINES
#include <cmath> 


c10::DeviceType device = c10::DeviceType::CPU;
const int max_points = 32;
const int max_voxels = 16000;
const int num_classes = 1;
const int box_code_size = 7;
const int nms_pre = 100;
const int max_num = 50;
const float score_thr = 0.1;
const float nms_thr = 0.01;
std::vector<float> voxel_size = { 0.16, 0.16, 4 };
std::vector<float> coors_range = { 0, -39.68, -3, 69.12, 39.68, 1 };
std::vector<int> feature_size = { 1, 248, 216 };

struct Point
{
	float x, y, z, r;
};


at::Tensor grid_anchors()
{
	at::Tensor anchor_range = torch::tensor(std::vector<float>{0, -39.68, -1.78, 69.12, 39.68, -1.78});
	at::Tensor x_centers = torch::linspace(anchor_range[0].item().toFloat(), anchor_range[3].item().toFloat(), feature_size[2] + 1).to(device);
	at::Tensor y_centers = torch::linspace(anchor_range[1].item().toFloat(), anchor_range[4].item().toFloat(), feature_size[1] + 1).to(device);
	at::Tensor z_centers = torch::linspace(anchor_range[2].item().toFloat(), anchor_range[5].item().toFloat(), feature_size[0] + 1).to(device);
	at::Tensor sizes = torch::tensor(std::vector<float>{ 3.9, 1.6, 1.56 }).to(device).reshape({ -1, 3 });
	at::Tensor rotations = torch::tensor(std::vector<float>{ 0, 1.5707963 }).to(device);
	at::Tensor x_shift = (x_centers[1] - x_centers[0]) / 2;
	at::Tensor y_shift = (y_centers[1] - y_centers[0]) / 2;
	at::Tensor z_shift = (z_centers[1] - z_centers[0]) / 2;
	x_centers += x_shift;
	y_centers += y_shift;
	z_centers += z_shift;

	std::vector<at::Tensor> rets = torch::meshgrid({ x_centers.slice(0, 0, feature_size[2]), y_centers.slice(0, 0, feature_size[1]), z_centers.slice(0, 0, feature_size[0]), rotations});
	for (size_t i = 0; i < rets.size(); i++)
	{
		rets[i] = rets[i].unsqueeze(-2).repeat({ 1, 1, 1, 1, 1 }).unsqueeze(-1);
	}
	sizes = sizes.reshape({ 1, 1, 1, -1, 1, 3 });
	c10::IntArrayRef tile_size_shape = rets[0].sizes();
	sizes = sizes.repeat(tile_size_shape);
	rets.insert(rets.begin() + 3, sizes);
	at::Tensor anchors = torch::cat(rets, -1).permute({ 2, 1, 0, 3, 4, 5 });
	anchors = anchors.reshape({ -1, anchors.size(-1) });
	return anchors;
}


at::Tensor decode(at::Tensor& anchors, at::Tensor& deltas)
{
	auto anchors_split = torch::split(anchors, 1, -1);
	at::Tensor xa = anchors_split[0];
	at::Tensor ya = anchors_split[1];
	at::Tensor za = anchors_split[2];
	at::Tensor wa = anchors_split[3];
	at::Tensor la = anchors_split[4];
	at::Tensor ha = anchors_split[5];
	at::Tensor ra = anchors_split[6];

	auto deltas_split = torch::split(deltas, 1, -1);
	at::Tensor xt = deltas_split[0];
	at::Tensor yt = deltas_split[1];
	at::Tensor zt = deltas_split[2];
	at::Tensor wt = deltas_split[3];
	at::Tensor lt = deltas_split[4];
	at::Tensor ht = deltas_split[5];
	at::Tensor rt = deltas_split[6];

	za = za + ha / 2;
	at::Tensor diagonal = torch::sqrt(la * la + wa * wa);
	at::Tensor xg = xt * diagonal + xa;
	at::Tensor yg = yt * diagonal + ya;
	at::Tensor zg = zt * ha + za;
	at::Tensor lg = torch::exp(lt) * la;
	at::Tensor wg = torch::exp(wt) * wa;
	at::Tensor hg = torch::exp(ht) * ha;
	at::Tensor rg = rt + ra;
	zg = zg - hg / 2;
	return torch::cat({ xg, yg, zg, wg, lg, hg, rg }, -1);
}


at::Tensor xywhr2xyxyr(at::Tensor& boxes_xywhr)
{
	at::Tensor boxes = torch::zeros_like(boxes_xywhr);
	at::Tensor half_w = boxes_xywhr.select(1, 2) / 2;
	at::Tensor half_h = boxes_xywhr.select(1, 3) / 2;
	boxes.select(1, 0) = boxes_xywhr.select(1, 0) - half_w;
	boxes.select(1, 1) = boxes_xywhr.select(1, 1) - half_h;
	boxes.select(1, 2) = boxes_xywhr.select(1, 0) + half_w;
	boxes.select(1, 3) = boxes_xywhr.select(1, 1) + half_h;
	boxes.select(1, 4) = boxes_xywhr.select(1, 4);
	return boxes;
}
	

std::vector<at::Tensor> box3d_multiclass_nms(at::Tensor& mlvl_bboxes, at::Tensor& mlvl_bboxes_for_nms, at::Tensor& mlvl_scores, at::Tensor& mlvl_dir_scores)
{
	int num_classes = mlvl_scores.size(1) - 1;
	std::vector<at::Tensor> bboxes_list, scores_list, labels_list, dir_scores_list;
	for (size_t i = 0; i < num_classes; i++)
	{
		at::Tensor cls_inds = (mlvl_scores.select(1, i) > score_thr);
		if (!cls_inds.any().item().toBool())
			continue;
		at::Tensor _scores = mlvl_scores.select(1, i);
		at::Tensor _bboxes_for_nms = mlvl_bboxes_for_nms.to(torch::kCUDA);
		at::Tensor order = std::get<1>(_scores.sort(0, true));
		at::Tensor keep = torch::zeros(_bboxes_for_nms.size(0), torch::kLong);
		int num_out = nms_gpu(_bboxes_for_nms, keep, nms_thr, 0);
		at::Tensor selected = keep.slice(0, 0, num_out);
		bboxes_list.push_back(mlvl_bboxes.index_select(0, selected));
		scores_list.push_back(_scores.index_select(0, selected));
		at::Tensor cls_label = mlvl_bboxes.new_full((selected.numel()), (long)i, torch::kLong);
		labels_list.push_back(cls_label);
		dir_scores_list.push_back(mlvl_dir_scores.index_select(0, selected));
	}

	at::Tensor bboxes, labels, scores, dir_scores;
	if (!bboxes_list.empty())
	{
		for (size_t i = 0; i < bboxes_list.size(); i++)
		{
			if (i == 0)
			{
				bboxes = bboxes_list[i];
				labels = labels_list[i];
				scores = scores_list[i];
				dir_scores = dir_scores_list[i];
			}
			else
			{
				bboxes = torch::cat({ bboxes, bboxes_list[i] }, 0);
				labels = torch::cat({ labels, labels_list[i] }, 0);
				scores = torch::cat({ scores, scores_list[i] }, 0);
				dir_scores = torch::cat({ dir_scores, dir_scores_list[i] }, 0);
			}
		}
		if (bboxes.sizes()[0] > max_num)
		{
			at::Tensor inds = std::get<1>(scores.sort(0, true));
			inds = inds.slice(0, 0, max_num);
			bboxes = bboxes.index_select(0, inds);
			labels = labels.index_select(0, inds);
			scores = scores.index_select(0, inds);
			dir_scores = dir_scores.index_select(0, inds);
		}
	}
	else
	{
		bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)));
		scores = mlvl_scores.new_zeros((0));
		labels = mlvl_scores.new_zeros((0), torch::kLong);
		dir_scores = mlvl_scores.new_zeros((0));
	}

	return { bboxes, scores, labels, dir_scores };
}


int main()
{
	std::vector<Point> pts;
	float x, y, z, r;
	std::ifstream infile("points.txt");
	while (infile >> x >> y >> z >> r)
	{
		Point pt;
		pt.x = x;
		pt.y = y;
		pt.z = z;
		pt.r = r;
		pts.push_back(pt);
	}

	at::Tensor points = torch::from_blob(pts.data(), { (int)pts.size(), 4 }, torch::kFloat32);
	at::Tensor voxels = torch::zeros({ max_voxels, max_points, 4 }, torch::kFloat32);
	at::Tensor coors = torch::zeros({ max_voxels, 3 }, torch::kInt32);
	at::Tensor num_points_per_voxel = torch::zeros({ max_voxels }, torch::kInt32);
	int voxel_num = voxelization::hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels);
	at::Tensor res_voxels = voxels.slice(0, 0, voxel_num);
	at::Tensor res_num_points = num_points_per_voxel.slice(0, 0, voxel_num);
	at::Tensor res_coors = coors.slice(0, 0, voxel_num);

	torch::jit::script::Module module = torch::jit::load("pointpillars.pt");
	auto outputs = module.forward({ res_voxels, res_coors, res_num_points }).toTuple();

	at::Tensor cls_score_list = outputs->elements()[0].toTensor();
	at::Tensor bbox_pred_list = outputs->elements()[1].toTensor();
	at::Tensor dir_cls_pred_list = outputs->elements()[2].toTensor();
	at::Tensor dir_cls_pred = torch::squeeze(dir_cls_pred_list).permute({ 1, 2, 0 }).reshape({ -1, 2 });
	auto dir_cls_scores = std::get<1>(torch::max(dir_cls_pred, -1));
	at::Tensor cls_score = torch::squeeze(cls_score_list).permute({ 1, 2, 0 }).reshape({ -1, num_classes });
	at::Tensor scores = cls_score.sigmoid();
	at::Tensor bbox_pred = torch::squeeze(bbox_pred_list).permute({ 1, 2, 0 }).reshape({ -1, box_code_size });
	at::Tensor max_scores = std::get<0>(torch::max(scores, 1));
	at::Tensor topk_inds = std::get<1>(torch::topk(max_scores, nms_pre));
	at::Tensor priors = torch::index_select(grid_anchors(), 0, topk_inds);
	bbox_pred = torch::index_select(bbox_pred, 0, topk_inds);
	scores = torch::index_select(scores, 0, topk_inds);
	dir_cls_scores = torch::index_select(dir_cls_scores, 0, topk_inds);
	at::Tensor bboxes = decode(priors, bbox_pred);
	at::Tensor mlvl_bboxes_bev = torch::cat({ bboxes.slice(1, 0, 2), bboxes.slice(1, 3, 5), bboxes.slice(1, 5, 6) }, 1);
	at::Tensor mlvl_bboxes_for_nms = xywhr2xyxyr(mlvl_bboxes_bev);
	at::Tensor padding = torch::zeros({ scores.size(0), 1 });
	scores = torch::cat({ scores, padding }, 1);
	std::vector<at::Tensor> result = box3d_multiclass_nms(bboxes, mlvl_bboxes_for_nms, scores, dir_cls_scores);
	at::Tensor result_bboxes = result[0];
	at::Tensor result_scores = result[1];
	at::Tensor result_labels = result[2];
	at::Tensor result_dir_scores = result[3];

	if(result_bboxes.sizes()[0] > 0)
	{
		at::Tensor dir_rot = result_bboxes.select(1, 6) + M_PI / 2 - torch::floor(result_bboxes.select(1, 6) + M_PI / 2 / M_PI) * M_PI;
		result_bboxes.select(1, 6) = (dir_rot - M_PI / 2 + M_PI * result_dir_scores.to(result_bboxes.dtype()));
	}
	std::cout << result_bboxes << std::endl;
	std::cout << result_scores << std::endl;
	std::cout << result_labels << std::endl;

	return 0;
}