#include "Yolo.h"

#include <fstream>

#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>

#include "CudaKernel.h"

namespace yolo
{
	YoloV5::YoloV5()
	{
		logger_ = Logger();
		SetBoundingBoxColor();

		result_callback_ = nullptr;
	}

	bool YoloV5::LoadLabels(const char* label_path)
	{
		std::ifstream ifs(label_path, std::ios::in);
		if (!ifs.good())
		{
			// Logger(Warning, "Open labels file failure");
			return false;
		}

		std::string str;
		while (std::getline(ifs, str))
		{
			size_t pos = str.find(":");
			assert(pos < str.size());

			int cls		= std::atoi(str.substr(0, pos).c_str());
			labels[cls] = str.substr(pos + 2, str.size());
		}
		return true;
	}

	void YoloV5::SetBoundingBoxColor()
	{
		for (int i = 0; i < num_labels_; ++i)
		{
			int r = std::rand() % 256;
			int g = std::rand() % 256;
			int b = std::rand() % 256;

			cv::Scalar color = cv::Scalar(r, g, b);
			bbox_colors[i]	 = color;
		}
	}

	int YoloV5::Run(const char* engine_path, const char* input_path, InputType type)
	{
		// Load engine model.
		nvinfer1::IRuntime*	   runtime = nvinfer1::createInferRuntime(logger_);
		nvinfer1::ICudaEngine* engine  = nullptr;
		if (!LoadEngine(engine_path, runtime, engine))
		{
			// Logger.
			return -1;
		}

		// Malloc Cuda memory for model input and output.
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		assert(engine->getNbBindings() == 2);
		size_t input_Bytes	= GetDimensionsSize(engine->getBindingDimensions(0)) * sizeof(float);
		size_t output_Bytes = GetDimensionsSize(engine->getBindingDimensions(1)) * sizeof(float);
		assert((input_Bytes == 3 * inference_size_ * inference_size_ * sizeof(float)) &&
			   (output_Bytes == 25200 * 85 * sizeof(float)));
		float* buffers[2] = {nullptr, nullptr};
		cudaMalloc((void**)&buffers[0], input_Bytes);
		cudaMalloc((void**)&buffers[1], output_Bytes);

		// Run
		stop_flag_ = false;
		switch (type)
		{
		case yolo::InputType::Camera:
			RunCamera(engine, buffers, input_Bytes, output_Bytes, stream);
			break;
		case yolo::InputType::Video:
			break;
		case yolo::InputType::Image:
			break;
		default:
			break;
		}

		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		engine->destroy();
		runtime->destroy();
		return 0;
	}

	void YoloV5::RunCamera(
		nvinfer1::ICudaEngine* engine, float** buffers, size_t input_size, size_t output_size, cudaStream_t& stream)
	{
		cv::VideoCapture capture(0);
		capture.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, raw_width_);
		capture.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, raw_height_);

		nvinfer1::IExecutionContext* context = engine->createExecutionContext();

		// Malloc memory include frame array, input and output.
		size_t		   frame_Bytes	= static_cast<size_t>(raw_height_ * raw_width_ * 3) * sizeof(unsigned char);
		unsigned char* frame_device = nullptr;
		cudaMalloc((void**)&frame_device, frame_Bytes);

		// Affine matrix.
		AffineMatrix affine		   = AffineMatrix(raw_height_, raw_width_, inference_size_, inference_size_);
		size_t		 affine_Bytes  = 6 * sizeof(float);
		float*		 affine_device = nullptr;
		cudaMalloc((void**)&affine_device, affine_Bytes);
		cudaMemcpyAsync(affine_device, affine.data, affine_Bytes, cudaMemcpyHostToDevice, stream);

		// Read frame and inference.
		std::vector<BoundingBox> bounding_boxes;
		float*					 output = new float[output_size];
		try
		{
			while (!stop_flag_)
			{
				capture >> frame;
				cv::flip(frame, frame, 1);

				// Convert image matrix to float array, then inference.
				cudaMemcpyAsync(frame_device, frame.data, frame_Bytes, cudaMemcpyHostToDevice, stream);
				cuda::ConvertMatToFloatArray(frame_device, raw_height_, raw_width_, 3, buffers[0], inference_size_,
					inference_size_, affine_device, 0, stream);

				context->executeV2((void**)buffers);
				cudaMemcpyAsync(output, buffers[1], output_size, cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);

				// Parser output.
				ParseResults(output, bounding_boxes);
				Show(bounding_boxes);

				cv::waitKey(1);
				bounding_boxes.clear();
			}
		}
		catch (std::exception& ex)
		{
			// Do nothing.
		}

		// Release resources.
		delete[] output;
		cudaFree(affine_device);
		cudaFree(frame_device);
		context->destroy();
	}

	size_t YoloV5::GetDimensionsSize(nvinfer1::Dims dims)
	{
		size_t size = 1;
		for (size_t i = 0; i < dims.nbDims; ++i)
		{
			size *= dims.d[i];
		}
		return size;
	}

	void YoloV5::RegisterCallback(ReadResultCallback callback)
	{
		if (callback != nullptr)
		{
			result_callback_ = callback;
		}
	}

	bool YoloV5::OnnxToEngine(const char* onnx_path, const char* engine_path)
	{
		Logger logger;

		nvinfer1::IBuilder*			  builder = nvinfer1::createInferBuilder(logger);
		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);

		// Parse onnx model.
		nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

		bool b_parser = parser->parseFromFile(onnx_path, static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
		if (!b_parser)
		{
			std::cerr << "[Builder] Open onnx model failure.." << std::endl;
		}
		for (int32_t i = 0; i < parser->getNbErrors(); ++i)
		{
			std::cerr << parser->getError(i)->desc() << std::endl;
		}

		nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
		config->setMaxWorkspaceSize(1ULL << 30);
		if (builder->platformHasFastFp16())
		{
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}

		// Serialized network.
		nvinfer1::IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
		std::ofstream		   engine_file(engine_path, std::ios::out | std::ios::binary);
		engine_file.write(static_cast<const char*>(engine->data()), engine->size());
		if (!engine_file.good())
		{
			std::cerr << "[Builder] Build engine file failure." << std::endl;
			return false;
		}
		engine_file.close();

		// Destory.
		engine->destroy();
		config->destroy();
		network->destroy();
		parser->destroy();
		builder->destroy();
		return false;
	}

	bool YoloV5::LoadEngine(const char* path, nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine)
	{
		std::ifstream engine_file(path, std::ios::in | std::ios::binary);
		if (!engine_file.good())
		{
			// std::cerr << "[Error] Cannot open engine file." << std::endl;
			return false;
		}

		engine_file.seekg(0, engine_file.end);
		size_t size = engine_file.tellg();
		engine_file.seekg(0, engine_file.beg);

		char* model_data = new char[size * sizeof(char)];
		engine_file.read(model_data, size);
		engine_file.close();

		engine = runtime->deserializeCudaEngine(model_data, size * sizeof(char));

		delete[] model_data;
		return true;
	}

	void YoloV5::ParseResults(const float* results, std::vector<BoundingBox>& boxes)
	{
		const float* ptr = results;
		for (int i = 0; i < 25200; ++i)
		{
			float objectness = *(ptr + 4);
			if (objectness >= min_objectness_)
			{
				int	  label		 = std::max_element(ptr + 5, ptr + 85) - (ptr + 5);
				float confidence = *(ptr + 5 + label);
				if (confidence >= min_confidence_)
				{
					float bx = *ptr;
					float by = *(ptr + 1);
					float bw = *(ptr + 2);
					float bh = *(ptr + 3);

					BoundingBox box;
					box.x1		   = bx - bw / 2;
					box.y1		   = by - bh / 2;
					box.x2		   = bx + bw / 2;
					box.y2		   = by + bh / 2;
					box.label	   = label;
					box.confidence = confidence;

					bbox_mutex_.lock();
					boxes.emplace_back(std::move(box));
					bbox_mutex_.unlock();
				}
			}
			ptr += 85;
		}

		NMS(boxes, iou_threshold_);
	}

	void YoloV5::Show(const std::vector<BoundingBox>& boxes)
	{
		float scale = std::max(frame.rows, frame.cols) / (float)inference_size_;

		float dw = inference_size_ - frame.cols / scale;
		float dh = inference_size_ - frame.rows / scale;

		for (const BoundingBox& box : boxes)
		{
			int x1 = static_cast<int>((box.x1 - dw / 2.0f) * scale);
			int y1 = static_cast<int>((box.y1 - dh / 2.0f) * scale);
			int x2 = static_cast<int>((box.x2 - dw / 2.0f) * scale);
			int y2 = static_cast<int>((box.y2 - dh / 2.0f) * scale);

			cv::Point p1 = cv::Point(std::max(0, x1), std::max(0, y1));
			cv::Point p2 = cv::Point(std::min(frame.cols, x2), std::min(frame.rows, y2));

			cv::rectangle(frame, cv::Rect(p1, p2), bbox_colors[box.label], 2);
			std::string text = labels[box.label] + " " + std::to_string(box.confidence);
			cv::putText(frame, text, p1, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.7, bbox_colors[box.label], 2);
		}

		if (result_callback_ != nullptr)
		{
			result_callback_(frame.data, frame.rows, frame.cols);
		}
	}

	float YoloV5::IOU(const BoundingBox& box1, const BoundingBox& box2)
	{
		float x1 = std::max(box1.x1, box2.x1);
		float y1 = std::max(box1.y1, box2.y1);
		float x2 = std::min(box1.x2, box2.x2);
		float y2 = std::min(box1.y2, box2.y2);

		if (x1 < x2 && y1 < y2)
		{
			float over_area = (x2 - x1) * (y2 - y1);
			float area_1	= (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
			float area_2	= (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
			return over_area / (area_1 + area_2 - over_area);
		}
		return 0.0f;
	}

	void YoloV5::NMS(std::vector<BoundingBox>& boxes, float threshold)
	{
		std::sort(boxes.begin(), boxes.end(),				   //
			[](const BoundingBox& lhs, const BoundingBox& rhs) //
			{												   //
				return lhs.confidence > lhs.confidence;		   //
			});

		for (auto iter = boxes.begin(); iter != boxes.end();)
		{
			for (auto other = iter + 1; other != boxes.end();)
			{
				other =
					(iter->label == other->label && IOU(*other, *iter) > threshold) ? boxes.erase(other) : other + 1;
			}
			++iter;
		}
	}

} // namespace yolo
