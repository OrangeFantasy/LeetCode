#pragma once

#include <cuda_runtime_api.h>

namespace cuda
{
	void ConvertMatToFloatArray(const unsigned char* src, int src_rows, int src_cols, int src_ch, //
		float* dst, int dst_rows, int dst_cols, float* affine, unsigned char border_value, const cudaStream_t& stream);

} // namespace cuda