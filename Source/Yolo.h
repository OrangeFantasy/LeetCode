#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "Logger.h"
#include "Utils.h"

namespace yolo
{
	enum InputType
	{
		Camera,
		Video,
		Image,
	};

	class YoloV5
	{
	public:
		static YoloV5& Instance();

		bool LoadLabels(const char* labels_path);
		int	 Run(const char* engine_path, const char* input_path = "", InputType type = InputType::Camera);
		void StopRun() { stop_flag_ = true; };

		using ReadResultCallback = bool (*)(unsigned char* image_data, int height, int width);
		void RegisterCallback(ReadResultCallback callback);

	private:
		YoloV5();
		YoloV5(const YoloV5&) = delete;
		YoloV5(YoloV5&&)	  = delete;

		void SetBoundingBoxColor();

		bool   OnnxToEngine(const char* onnx_path, const char* engin_path);
		bool   LoadEngine(const char* path, nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine);
		size_t GetDimensionsSize(nvinfer1::Dims dims);

		void ParseResults(const float* results, std::vector<BoundingBox>& boxes);
		void Show(const std::vector<BoundingBox>& boxes);

		float IOU(const BoundingBox& box1, const BoundingBox& box2);
		void  NMS(std::vector<BoundingBox>& boxes, float threshold);

		void RunCamera(nvinfer1::ICudaEngine* engine, float** buffers, size_t input_size, size_t output_size, cudaStream_t& stream);

	private:
		int raw_width_	= 1280;
		int raw_height_ = 720;

		int	  inference_size_ = 640;
		int	  num_labels_	  = 80;
		float min_objectness_ = 0.45f;
		float min_confidence_ = 0.25f;
		float iou_threshold_  = 0.5f;

		bool stop_flag_ = false;

		std::unordered_map<int, std::string> labels;
		std::unordered_map<int, cv::Scalar>	 bbox_colors;

		Logger	   logger_;
		std::mutex bbox_mutex_;

		cv::Mat			   frame;
		ReadResultCallback result_callback_;
	};

	inline YoloV5& yolo::YoloV5::Instance()
	{
		static yolo::YoloV5 unique_instance = yolo::YoloV5();
		return unique_instance;
	}
} // namespace yolo