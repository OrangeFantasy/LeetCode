#ifndef __EXPORT__H__
#define __EXPORT__H__

#include "Yolo.h"

#define DLLAPI extern "C" __declspec(dllexport)

#define Model yolo::YoloV5::Instance()

typedef bool (*ReadImageDelegate)(unsigned char* image_data, int height, int width);

DLLAPI bool LoadLabels(const char* labels_path);

DLLAPI bool RunModel(const char* engine_path, const char* input_path, int type);

DLLAPI void StopRun();

DLLAPI void RegisterReadImageDelegate(ReadImageDelegate callback);

#endif // !__EXPORT__H__
