#include "Utils.h"

#include <utility>

yolo::AffineMatrix::AffineMatrix(int src_rows, int src_cols, int dst_rows, int dst_cols)
{

	float scale_x = src_cols / (float)dst_cols;
	float scale_y = src_rows / (float)dst_rows;

	float scale = std::max(scale_x, scale_y);

	data[0] = scale;
	data[1] = 0.0f;
	data[2] = src_cols * 0.5f - scale * dst_cols * 0.5f - 0.5f;
	data[3] = 0.0f;
	data[4] = scale;
	data[5] = src_rows * 0.5f - scale * dst_rows * 0.5f - 0.5f;
}