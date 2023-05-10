#pragma once

#include <NvInfer.h>

class Logger : public nvinfer1::ILogger
{
public:
	explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING);

	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

private:
	nvinfer1::ILogger::Severity severity_;
};