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

#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "dnhpx_face_gender.h"
#include "dnhpx_face_normalization.h"
#include "dnhpx_auto_array.h"
#include "dnhpx_error_code.h"


#ifndef _WIN32
#define _MAX_PATH 260
#endif

#define FACE_GENDER_MODEL_NAME "librls.so"
#define FACE_GENDER_PARAM_NAME "librls.param"
#define FACE_GENDER_BIN_NAME "librls.bin"

namespace
{
    char g_szDeepFeatSDKPath[_MAX_PATH] = { 0 };
    int g_num_threads = 1;
    bool g_light_mode = true;
    dnhpx::CNormImage3pt affineNorm;
    dnhpx::AutoArray<unsigned char> pWeightBuf;
    const float mean_vals[3] = { 127.5, 127.5, 127.5 };
    const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };

    volatile bool g_bFaceGenderInited = false;
    volatile int g_FaceGenderInitCount = 0;
}

int __stdcall DNHPXGetFaceGenderScore(DNHPXFaceAttHandle handle,
    const DNHPXPointF*feaPoints, const unsigned char *image_data, int width,
    int height, int channel, float &gender_score, int &age, float &beauty_score,
    float &glass_score, int &emotion, float &happy_score)
{
    if (image_data == 0 || width <= 0 || height <= 0)
        return  DNHPX_INVALID_INPUT;

    if (1 != channel && 3 != channel)
        return DNHPX_INVALID_IMAGE_FORMAT;

    if (!g_bFaceGenderInited)
        return DNHPX_MODEL_NOT_INITIALIZED;

    int nRet = 0;
    //clock_t count;
    try
    {
        ncnn::Mat ncnn_face_img(96, 128, 3, 4u);
        if (ncnn_face_img.empty())
            return DNHPX_MEMORY_ALLOC_ERROR;

        dnhpx::AutoArray<float> feature_points_5(10);
        for (int i = 0; i < 5; ++i) {
            feature_points_5[2 * i] = feaPoints[i].x;
            feature_points_5[2 * i + 1] = feaPoints[i].y;
        }

        //count = clock();
        int size = width * height;
        if (channel == 3) {
            affineNorm.NormImageRaw2Planar(image_data, width, height,
                channel, feature_points_5, 5, (float *)ncnn_face_img);

            ncnn_face_img.substract_mean_normalize(mean_vals, norm_vals);
        }
        else {
            float* ptr0 = ncnn_face_img.channel(0);
            float* ptr1 = ncnn_face_img.channel(1);
            float* ptr2 = ncnn_face_img.channel(2);
            affineNorm.NormImage(
                image_data, //pImage.begin() + j * testImg.rows * testImg.cols,
                width, height, feature_points_5, 5,
                ptr0);
            memcpy(ptr1, ptr0, sizeof(float) * 12288);
            memcpy(ptr2, ptr0, sizeof(float) * 12288);

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
        ex.extract("output", out);

        //count = clock() - count;
        //std::cout << "Feature: " << count << std::endl;

        //count = clock();
        //AutoArray<float> pFeatures(out.total());
        //for (int j = 0; j<out.total(); j++)
        /*{
            pFeatures[j] = out[j];
        }*/

        gender_score = out[0];
        glass_score = out[2];
        
        float emotion_vec[7];
        float max_score = -10000.0f;
        float min_score = 10000.0f;
        for (int c = 0; c < 7; ++c) {
            emotion_vec[c] = out[c + 5];
            if (max_score < out[c + 5]) {
                max_score = out[c + 5];
                emotion = c;
            }  
            if (min_score > out[c + 5]) {
                min_score = out[c + 5];
            }
        }
        //happy_score = (happy_score - min_score) / (max_score - min_score);
        age = 0;
        int iter = (out.total() - 14) / 2;
        for (int c = 0; c < iter; ++c)
        {
            if (out[2 * c + 12] < out[2 * c + 13])
                age++;
        }
        age = age + 16;

        beauty_score = out[out.total() - 2];
        happy_score = out[out.total() - 1];
        //count = clock() - count;
        //std::cout << "result: " << count << std::endl;
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


int __stdcall DNHPXSetFaceGenderLibPath(const char *szLibPath)
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

int __stdcall DNHPXInitFaceGender(const char *szNetName,
    DNHPXFaceAttHandle* pHandle, int num_threads, bool light_mode)
{
	if (pHandle == NULL)
		return -1;
	
	// initialize deep face network
	*pHandle = NULL;
	std::locale::global(std::locale(""));

    if (g_bFaceGenderInited)
    {
        ++g_FaceGenderInitCount;
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
            strDllPath += FACE_GENDER_MODEL_NAME;

        dnhpx::CAlgorithmDomain* pCaffeNet = new dnhpx::CAlgorithmDomain();
        pCaffeNet->init(strDllPath.c_str());

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        float NormPoints_128[10] = {
            35.5f, 55.48f,
            91.5f, 55.48f,
            63.5f, 83.98f,
            45.5f, 111.48f,
            81.5f, 111.48f,
        };
        affineNorm.Initialize(96, 128, 0.78125, 128, NormPoints_128);

		*pHandle = reinterpret_cast<DNHPXFaceAttHandle>(pCaffeNet);

        g_bFaceGenderInited = true;
        ++g_FaceGenderInitCount;
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

int __stdcall DNHPXInitOLDFaceGender(const char *szParamName,
    const char* szBinName, DNHPXFaceAttHandle* pHandle,
    int num_threads, bool light_mode)
{
    printf("该函数即将停止使用，请使用函数：DNHPXInitFaceGender\n");

    if (pHandle == NULL)
        return -1;

    // initialize deep face network
    *pHandle = NULL;
    std::locale::global(std::locale(""));

    if (g_bFaceGenderInited)
    {
        ++g_FaceGenderInitCount;
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
            strParamPath += FACE_GENDER_PARAM_NAME;

        if (szBinName != 0)
            strBinPath += szBinName;
        else
            strBinPath += FACE_GENDER_BIN_NAME;

        dnhpx::CAlgorithmDomain* pCaffeNet = new dnhpx::CAlgorithmDomain();
        pCaffeNet->init(std::vector<std::string>(1, strParamPath),
            std::vector<std::string>(1, strBinPath));
        /*pCaffeNet->load_param(strParamPath.c_str());
        pCaffeNet->load_model(strBinPath.c_str());*/

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        float NormPoints_128[10] = {
            35.5f, 55.48f,
            91.5f, 55.48f,
            63.5f, 83.98f,
            45.5f, 111.48f,
            81.5f, 111.48f,
        };
        affineNorm.Initialize(96, 128, 0.78125, 128, NormPoints_128);

        *pHandle = reinterpret_cast<DNHPXFaceAttHandle>(pCaffeNet);

        g_bFaceGenderInited = true;
        ++g_FaceGenderInitCount;
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

int __stdcall DNHPXUninitFaceGender(DNHPXFaceAttHandle handle)
{
    if (g_bFaceGenderInited) {
        --g_FaceGenderInitCount;
        if (g_FaceGenderInitCount == 0) {

            // 如果满足被释放的条件，则将初始化参数设置为false
            g_bFaceGenderInited = false;

            dnhpx::CAlgorithmDomain* pCaffeNet = reinterpret_cast<dnhpx::CAlgorithmDomain *>(handle);
            pCaffeNet->clear();
            delete pCaffeNet;
        }
    }
	return 0;
}
