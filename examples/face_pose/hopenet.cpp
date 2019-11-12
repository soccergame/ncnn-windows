#include "hopenet.h"
#include <iostream>
#include <string>
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

double postProcessing(std::vector<double> pred) {	
	std::vector<double>::iterator max = std::max_element(std::begin(pred), std::end(pred));
	double maxVal = *max;
	// std::cout << "maxVal " << maxVal << std::endl;
	std::vector<double> softmaxPred;
	double sumTmp= 0.0;
	for (int i=0; i < 66; ++i)
	{
		double tmp = exp(pred[i] - maxVal);
		sumTmp += tmp;
		softmaxPred.push_back(tmp);		
	}

	const double NEAR_0 = 1e-10;
	for (int i = 0; i < 66; ++i)
	{
		softmaxPred[i] = softmaxPred[i] / sumTmp + NEAR_0;
		// std::cout << softmaxPred[i] << std::endl;
	}

	double sumExpectation = 0.0;
	for (int i = 0; i < 66; ++i)
	{
		sumExpectation += softmaxPred[i] * (double)(i + 1.0);
	}

	return sumExpectation * 3.0 - 99.0;
}


Hopenet::Hopenet() {
	hopenet_ = new ncnn::Net();
	initialized = false;
}

Hopenet::~Hopenet() {
	hopenet_->clear();
}

int Hopenet::LoadModel(const char * root_path) {
	std::string hopenet_param = std::string(root_path) + "/hopenet.param";
	std::string hopenet_bin = std::string(root_path) + "/hopenet.bin";
	if (hopenet_->load_param(hopenet_param.c_str()) == -1 ||
		hopenet_->load_model(hopenet_bin.c_str()) == -1) {
		std::cout << "load hopenet model failed." << std::endl;
		return 10000;
	}
	initialized = true;
	return 0;
}


int Hopenet::ExtractHeadpose(const cv::Mat & img_src,
	const cv::Rect & face, std::vector<double>* headpose) {
	std::cout << "start extract headpose." << std::endl;
	headpose->clear();
	if (!initialized) {
		std::cout << "headpose unitialized." << std::endl;
		return 10000;
	}

	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	cv::Mat img_face_bgr = img_src(face).clone();
	// cv::imwrite("../images/cropped.jpg", img_face_bgr);
	// cv::Vec3b color_value = img_face_bgr.at<cv::Vec3b>(0, 0);
	// std::cout << "Vec3b(0, 0) " << color_value << std::endl;
	
	ncnn::Mat img_face_gray = ncnn::Mat::from_pixels(img_face_bgr.data,
		ncnn::Mat::PIXEL_BGR2GRAY, img_face_bgr.cols, img_face_bgr.rows);
	ncnn::Mat in;
	ncnn::resize_bilinear(img_face_gray, in, 48, 48);

	// normalization
	// in.substract_mean_normalize(meanVals, normVals);

	ncnn::Extractor ex = hopenet_->create_extractor();
	ex.input("data", in);	
	ncnn::Mat out;

	ex.extract("hybridsequential0_multitask0_dense0_fwd", out); // final output layer
	
	// post processing	
	std::vector<double> predPitch, predRoll, predYaw;
	for (int i=0; i<66; ++i){
		predPitch.push_back(out[i]);
		// std::cout << out[i] << std::endl;
	}
	for (int i=66; i<132; ++i){
		predRoll.push_back(out[i]);
	}
	for (int i=132; i<198; ++i){
		predYaw.push_back(out[i]);
	}
	
	// pitch times -1 to keep consistence with other api
	double pitch = postProcessing(predPitch) * (-1.0);
	double roll = postProcessing(predRoll);
	double yaw = postProcessing(predYaw);

	// std::cout << "pitch yaw roll: " << pitch << " " << yaw << " " << roll << std::endl;

	headpose->push_back(pitch);
	headpose->push_back(yaw);
	headpose->push_back(roll);
	
	std::cout << "end extract headpose." << std::endl;

	return 0;
}
