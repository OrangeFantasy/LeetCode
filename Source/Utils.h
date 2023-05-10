#pragma once

namespace yolo
{
	struct AffineMatrix
	{
		AffineMatrix(int src_rows, int src_cols, int dst_rows, int dst_cols);

		float data[6] = {0.0f};
	};

	struct BoundingBox
	{
		float x1;
		float y1;
		float x2;
		float y2;
		float confidence;
		int	  label;
	};
} // namespace yolo