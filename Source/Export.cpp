#include "Export.h"

DLLAPI bool LoadLabels(const char* labels_path)
{
	return Model.LoadLabels(labels_path);
}

DLLAPI bool RunModel(const char* engine_path, const char* input_path, int type)
{
	try
	{
		Model.Run(engine_path, input_path, static_cast<yolo::InputType>(type));
	}
	catch (std::exception& ex)
	{
		return false;
	}
	return true;
}

DLLAPI void StopRun()
{
	Model.StopRun();
}

DLLAPI void RegisterReadImageDelegate(ReadImageDelegate callback)
{
	Model.RegisterCallback(callback);
}

//int main()
//{
//	const char* engine_path = R"(E:\_Project\Pytorch\yolov5\yolov5s.engine)";
//	const char* label_path	= R"(E:\_Project\_\TensorRT\classes.txt)";
//
//	yolo::YoloV5 Yolo;
//	Yolo.LoadLabels(label_path);
//	Yolo.Run(engine_path, yolo::InputType::Camera);
//
//	return 0;
//}