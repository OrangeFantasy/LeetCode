#include "Logger.h"

#include <iostream>

Logger::Logger(nvinfer1::ILogger::Severity severity)
	:severity_(severity)
{
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
	if (severity <= severity_)
	{
		std::cerr << msg << std::endl;
	}
}
