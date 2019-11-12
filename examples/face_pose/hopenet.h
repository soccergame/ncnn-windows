#ifndef _FACE_HOPENET_H_
#define _FACE_HOPENET_H_

#include "../headpose.h"
#include "ncnn/net.h"

class Hopenet : public Headpose {
public:
	Hopenet();
	~Hopenet();

	int LoadModel(const char* root_path);
	int ExtractHeadpose(const cv::Mat& img_src,
		const cv::Rect& face, std::vector<double>* headpose);

private:
	ncnn::Net* hopenet_;
	const float meanVals[1] = { 127.5f};
	const float normVals[1] = { 0.0078125f};
	bool initialized;
};

#endif // !_FACE_HOPENET_H_
