#pragma once
#include <torch/all.h>

int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id);


