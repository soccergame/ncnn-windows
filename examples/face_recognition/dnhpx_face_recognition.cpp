// THIDFaceDeepFeat.cpp : Defines the exported functions for the DLL application.
//

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <time.h>
#include <iostream>
#include "dnhpx_algorithm_utils.h"
#include "dnhpx_face_normalization.h"
#include "dnhpx_error_code.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "dnhpx_face_recognition.h"

#ifndef _WIN32
#define _MAX_PATH 260
#endif

#define FACE_RECOG_MODEL_NAME "mobileFaceNet.so"
#define FACE_RECOG_PARAM_NAME "mobileFaceNet.param"
#define FACE_RECOG_BIN_NAME "mobileFaceNet.bin"

namespace
{
    char g_szDeepFeatSDKPath[_MAX_PATH] = { 0 };
    int g_num_threads = 1;
    bool g_light_mode = true;
    dnhpx::CNormImageSimilarity affineNorm;
    dnhpx::AutoArray<unsigned char> pWeightBuf;
    const float mean_vals[3] = { 127.5, 127.5, 127.5 };
    const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };

    volatile bool g_bFaceRecognitionInited = false;
    volatile int g_FaceRecognitionInitCount = 0;
}

int __stdcall GetFaceRecognitionFeature(DNHPXFaceRecogHandle handle,
    const float *feaPoints, const unsigned char *image_data, int width,
    int height, int channel, float **feature, int &fea_dim)
{
    if (image_data == 0 || width <= 0 || height <= 0)
        return  DNHPX_INVALID_INPUT;

    if (1 != channel && 3 != channel)
        return DNHPX_INVALID_IMAGE_FORMAT;

    if (!g_bFaceRecognitionInited)
        return DNHPX_MODEL_NOT_INITIALIZED;

    int nRet = 0;
    //clock_t count;
    try
    {
        ncnn::Mat ncnn_face_img(112, 112, 3, 4u);
        if (ncnn_face_img.empty())
            return DNHPX_INVALID_INPUT;

        //count = clock();
        int size = width * height;
        if (channel == 3) {
            affineNorm.NormImageRaw2Planar(image_data, width, height,
                channel, feaPoints, 5, (float *)ncnn_face_img);
            /*AutoArray<unsigned char> trans_image_data(112 * 112 * 3);
            affineNorm.NormImageRaw2Planar(image_data, width, height,
                channel, feaPoints, 5, trans_image_data.begin());
            cv::Mat img1 = cv::Mat(112, 112, CV_8UC1, trans_image_data.begin());
            cv::Mat img2 = cv::Mat(112, 112, CV_8UC1, trans_image_data.begin() + 112 * 112);
            cv::Mat img3 = cv::Mat(112, 112, CV_8UC1, trans_image_data.begin() + 112 * 112 * 2);
            std::vector<cv::Mat> img;
            img.push_back(img3);
            img.push_back(img2);
            img.push_back(img1);
            cv::Mat test_img;
            cv::merge(img, test_img);
            cv::imwrite("D:/project/hzx.jpg", test_img);*/
            //ncnn_face_img.substract_mean_normalize(mean_vals, norm_vals);
        }
        else {
            float* ptr0 = ncnn_face_img.channel(0);
            float* ptr1 = ncnn_face_img.channel(1);
            float* ptr2 = ncnn_face_img.channel(2);
            affineNorm.NormImage(
                image_data, //pImage.begin() + j * testImg.rows * testImg.cols,
                width, height, feaPoints, 5,
                ptr0);
            memcpy(ptr1, ptr0, sizeof(float) * 12544);
            memcpy(ptr2, ptr0, sizeof(float) * 12544);

            ncnn_face_img.substract_mean_normalize(mean_vals, norm_vals);
        }

        //count = clock() - count;
        //std::cout << "Norm: " << count << std::endl;

        dnhpx::CAlgorithmDomain* pCaffeNet = reinterpret_cast<dnhpx::CAlgorithmDomain*>(handle);
        ncnn::Extractor ex = pCaffeNet->get_model()->create_extractor();
        ex.set_light_mode(g_light_mode);
        ex.set_num_threads(g_num_threads);

        //count = clock();
        ex.input("data", ncnn_face_img);
        ncnn::Mat out;
        ex.extract("fc1", out);

        fea_dim = out.total();
        (*feature) = new float[out.total()];
        float sum = 0;
        for (int j = 0; j<out.total(); j++)
        {
            sum += out[j] * out[j];
        }
        sum = 1.0f / (sqrt(sum) + 1e-6);
        for (int j = 0; j<out.total(); j++)
        {
            (*feature)[j] = out[j] * sum;
        }
    }
    catch (const std::bad_alloc &)
    {
        nRet = -2;
    }
    catch (const int &errCode)
    {
        nRet = errCode;
    }
    catch (...)
    {
        nRet = -3;
    }

    return nRet;
}

int __stdcall GetFaceRecognitionFeatureRaw(DNHPXFaceRecogHandle handle,
    const unsigned char *norm_data, float **feature, int &fea_dim)
{
    if (norm_data == 0)
        return  DNHPX_INVALID_INPUT;

    if (!g_bFaceRecognitionInited)
        return DNHPX_MODEL_NOT_INITIALIZED;

    int nRet = 0;
    //clock_t count;
    try
    {
        ncnn::Mat ncnn_face_img = ncnn::Mat::from_pixels(
            norm_data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
        if (ncnn_face_img.empty())
            return DNHPX_INVALID_INPUT;

        dnhpx::CAlgorithmDomain* pCaffeNet = reinterpret_cast<dnhpx::CAlgorithmDomain*>(handle);
        ncnn::Extractor ex = pCaffeNet->get_model()->create_extractor();
        ex.set_light_mode(g_light_mode);
        ex.set_num_threads(g_num_threads);

        //count = clock();
        ex.input("data", ncnn_face_img);
        ncnn::Mat out;
        ex.extract("fc1", out);

        fea_dim = out.total();
        (*feature) = new float[out.total()];
        for (int j = 0; j<out.total(); j++)
        {
            (*feature)[j] = out[j];
        }
    }
    catch (const std::bad_alloc &)
    {
        nRet = -2;
    }
    catch (const int &errCode)
    {
        nRet = errCode;
    }
    catch (...)
    {
        nRet = -3;
    }

    return nRet;
}


int __stdcall SetFaceRecognitionLibPath(const char *szLibPath)
{
	if (szLibPath == NULL)
		return -1;

#ifdef _WIN32
	strcpy_s(g_szDeepFeatSDKPath, _MAX_PATH, szLibPath);
#else
    strncpy(g_szDeepFeatSDKPath, szLibPath, _MAX_PATH);
#endif
	
	size_t len = strlen(g_szDeepFeatSDKPath);
	if (len != 0)
	{
	#ifdef _WIN32
		if (g_szDeepFeatSDKPath[len - 1] != '\\')
			strcat_s(g_szDeepFeatSDKPath, "\\");
	#else
	    if (g_szDeepFeatSDKPath[len - 1] != '/')
	        strncat(g_szDeepFeatSDKPath, "/", _MAX_PATH);
	#endif
	}

	return 0;
}

int __stdcall InitFaceRecognition(const char *szNetName,
    DNHPXFaceRecogHandle*pHandle, int num_threads, bool light_mode)
{
	if (pHandle == NULL)
		return -1;
	
	// initialize deep face network
	*pHandle = NULL;
	std::locale::global(std::locale(""));

    if (g_bFaceRecognitionInited)
    {
        ++g_FaceRecognitionInitCount;
        return DNHPX_OK;
    }

	int retValue = 0;

#ifndef _WIN32	
	if (strlen(g_szDeepFeatSDKPath) == 0)
		strncpy(g_szDeepFeatSDKPath, "./", _MAX_PATH);
#endif

	try
	{
#ifndef OLD_NCNN
		std::string strDllPath;
		strDllPath = g_szDeepFeatSDKPath;
        if (szNetName != 0)
            strDllPath += szNetName;
        else
            strDllPath += FACE_RECOG_MODEL_NAME;

        dnhpx::CAlgorithmDomain* pCaffeNet = new dnhpx::CAlgorithmDomain();
        pCaffeNet->init(strDllPath.c_str());

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        float NormPoints_128[10] = {
            38.2946f, 51.6963f,
            73.5318f, 51.5014f,
            56.0252f, 71.7366f,
            41.5493f, 92.3655f,
            70.7299f, 92.2041f,
        };
        /*affineNorm.Initialize(96, 128, 0.78125, 128, NormPoints_128);*/
        affineNorm.Initialize(112, 112, 1.0, 112, NormPoints_128);

		*pHandle = reinterpret_cast<DNHPXFaceRecogHandle>(pCaffeNet);

        g_bFaceRecognitionInited = true;
        ++g_FaceRecognitionInitCount;
#endif
	}
	catch (const std::bad_alloc &)
	{
		retValue = -2;
	}
	catch (const int &errCode)
	{
		retValue = errCode;
	}
	catch (...)
	{
		retValue = -3;
	}

	return retValue;
}

int __stdcall InitOLDFaceRecognition(const char *szParamName,
    const char *szBinName, DNHPXFaceRecogHandle*pHandle,
    int num_threads, bool light_mode)
{
    if (pHandle == NULL)
        return -1;

    // initialize deep face network
    *pHandle = NULL;
    std::locale::global(std::locale(""));

    if (g_bFaceRecognitionInited)
    {
        ++g_FaceRecognitionInitCount;
        return DNHPX_OK;
    }

    int retValue = 0;

#ifndef _WIN32	
    if (strlen(g_szDeepFeatSDKPath) == 0)
        strncpy(g_szDeepFeatSDKPath, "./", _MAX_PATH);
#endif

    try
    {
        std::string strParamPath, strBinPath;
        strParamPath = g_szDeepFeatSDKPath;
        strBinPath = g_szDeepFeatSDKPath;
        if (szParamName != 0)
            strParamPath += szParamName;
        else
            strParamPath += FACE_RECOG_PARAM_NAME;

        if (szBinName != 0)
            strBinPath += szBinName;
        else
            strBinPath += FACE_RECOG_BIN_NAME;

        dnhpx::CAlgorithmDomain* pCaffeNet = new dnhpx::CAlgorithmDomain();
        pCaffeNet->init(std::vector<std::string>(1, strParamPath),
            std::vector<std::string>(1, strBinPath));

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        float NormPoints_128[10] = {
            38.2946f, 51.6963f,
            73.5318f, 51.5014f,
            56.0252f, 71.7366f,
            41.5493f, 92.3655f,
            70.7299f, 92.2041f,
        };
        affineNorm.Initialize(112, 112, 1.0, 112, NormPoints_128);

        *pHandle = reinterpret_cast<DNHPXFaceRecogHandle>(pCaffeNet);

        g_bFaceRecognitionInited = true;
        ++g_FaceRecognitionInitCount;
    }
    catch (const std::bad_alloc &)
    {
        retValue = -2;
    }
    catch (const int &errCode)
    {
        retValue = errCode;
    }
    catch (...)
    {
        retValue = -3;
    }

    return retValue;
}

int __stdcall UninitFaceRecognition(DNHPXFaceRecogHandle handle)
{
    if (g_bFaceRecognitionInited) {
        --g_FaceRecognitionInitCount;
        if (g_FaceRecognitionInitCount == 0) {

            // 初始化变量修改
            g_bFaceRecognitionInited = false;

            dnhpx::CAlgorithmDomain* pCaffeNet = reinterpret_cast<dnhpx::CAlgorithmDomain*>(handle);
            pCaffeNet->clear();
            delete pCaffeNet;
        }
    }
	return 0;
}
