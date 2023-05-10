#include "CudaKernel.h"

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <device_launch_parameters.h>

constexpr int gpu_threads = 1024;

namespace cuda
{
	__global__ void ConvertMatToFloatArrayKernel(const unsigned char* src, int src_rows, int src_cols, int src_ch, //
		float* dst, int dst_rows, int dst_cols, const float* affine, unsigned char border_value)
	{
		// Current thread index and current pixel position.
		int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
		int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
		if (pos_x >= dst_cols || pos_y >= dst_rows)
		{
			return;
		}

		// Caclute source pixel position.
		int src_x = affine[0] * pos_x + affine[1] * pos_y + affine[2];
		int src_y = affine[3] * pos_x + affine[4] * pos_y + affine[5];

		// Bilinear interpolation.
		float value0 = border_value, value1 = border_value, value2 = border_value;
		if (src_x > 0.0f && src_x < src_cols - 1 && src_y > 0.0f && src_y < src_rows - 1)
		{
			// Interpolation weight (w_1, ..., w_4).
			int src_x1 = std::floorf(src_x);
			int src_y1 = std::floorf(src_y);
			int src_x2 = src_x1 + 1;
			int src_y2 = src_y1 + 1;

			float x_x1 = src_x - src_x1;
			float y_y1 = src_y - src_y1;
			float x2_x = src_x2 - src_x;
			float y2_y = src_y2 - src_y;

			float w1 = x2_x * y2_y;
			float w2 = x_x1 * y2_y;
			float w3 = x2_x * y_y1;
			float w4 = x_x1 * y_y1;

			// The last four points
			const unsigned char* p1 = src + src_y1 * src_cols * src_ch + src_x1 * src_ch;
			const unsigned char* p2 = src + src_y1 * src_cols * src_ch + src_x2 * src_ch;
			const unsigned char* p3 = src + src_y2 * src_cols * src_ch + src_x1 * src_ch;
			const unsigned char* p4 = src + src_y2 * src_cols * src_ch + src_x2 * src_ch;

			// Current pixel value.
			value0 = std::floorf(w1 * p1[0] + w2 * p2[0] + w3 * p3[0] + w4 * p4[0] + 0.5f);
			value1 = std::floorf(w1 * p1[1] + w2 * p2[1] + w3 * p3[1] + w4 * p4[1] + 0.5f);
			value2 = std::floorf(w1 * p1[2] + w2 * p2[2] + w3 * p3[2] + w4 * p4[2] + 0.5f);
		}
		int hw								   = dst_rows * dst_cols;
		dst[pos_y * dst_cols + pos_x]		   = value0 / 255.0f;
		dst[pos_y * dst_cols + pos_x + hw]	   = value1 / 255.0f;
		dst[pos_y * dst_cols + pos_x + 2 * hw] = value2 / 255.0f;
	}

	void ConvertMatToFloatArray(const unsigned char* src, int src_rows, int src_cols, int src_ch, //
		float* dst, int dst_rows, int dst_cols, float* affine, unsigned char border_value, const cudaStream_t& stream)
	{
		dim3 block(32, 32);
		dim3 grid((dst_cols + block.x - 1) / block.x, (dst_rows + block.y - 1) / block.y);

		ConvertMatToFloatArrayKernel<<<grid, block, 0, stream>>>(src, src_rows, src_cols, src_ch, dst, dst_rows, dst_cols, affine, border_value);
	}

} // namespace cuda
