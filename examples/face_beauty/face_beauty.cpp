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
#include "AlgorithmUtils.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "face_beauty.h"
#include "NormFaceImage.h"
#include "autoarray.h"
#include "Error_Code.h"
#include "net.h"


#ifndef _WIN32
#define _MAX_PATH 260
#endif

#define FACE_BEAUTY_MODEL_NAME "libsnfb.so"
#define FACE_BEAUTY_PARAM_NAME "libsnfb.param"
#define FACE_BEAUTY_BIN_NAME "libsnfb.bin"

namespace
{
    char g_szDeepFeatSDKPath[_MAX_PATH] = { 0 };
    int g_num_threads = 1;
    bool g_light_mode = true;
    hzx::CNormImageSimilarity affineNorm;
    AutoArray<unsigned char> pWeightBuf;
}

int __stdcall GetFaceBeautyScore(BeautyHandle handle, 
    const float *feaPoints, const unsigned char *image_data, int width,
    int height, int channel, float &beauty_score)
{
    if (image_data == 0 || width <= 0 || height <= 0)
        return  INVALID_IMAGE;

    if (1 != channel && 3 != channel)
        return INVALID_IMAGE_FORMAT;

    int nRet = 0;
    clock_t count;
    try
    {
        ncnn::Mat ncnn_face_img(256, 256, 3, 4u);
        if (ncnn_face_img.empty())
            return INVALID_IMAGE;

        int size = width * height;
        if (channel == 3) {
            affineNorm.NormImageRaw2Planar(
                image_data, width, height, channel,
                feaPoints, 5, (float *)ncnn_face_img
            );
        }
        else {
            float* ptr0 = ncnn_face_img.channel(0);
            float* ptr1 = ncnn_face_img.channel(1);
            float* ptr2 = ncnn_face_img.channel(2);
            affineNorm.NormImage(
                image_data,
                width, height, feaPoints, 5,
                ptr0);
            memcpy(ptr1, ptr0, sizeof(float) * 65535);
            memcpy(ptr2, ptr0, sizeof(float) * 65535);
        }

        ncnn::Net *pCaffeNet = reinterpret_cast<ncnn::Net *>(handle);
        ncnn::Extractor ex = pCaffeNet->create_extractor();
        ex.set_light_mode(g_light_mode);
        ex.set_num_threads(g_num_threads);

        //count = clock();
        ex.input("data", ncnn_face_img);
        ncnn::Mat out;
        ex.extract("fc7", out);
        
        beauty_score = out[0];
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


int __stdcall SetFaceBeautyLibPath(const char *szLibPath)
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

int __stdcall InitFaceBeauty(const char *szNetName,
    BeautyHandle *pHandle, int num_threads, bool light_mode)
{
	if (pHandle == NULL)
		return -1;
	
	// initialize deep face network
	*pHandle = NULL;
	std::locale::global(std::locale(""));

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
            strDllPath += FACE_BEAUTY_MODEL_NAME;

        std::fstream fileModel;
        fileModel.open(strDllPath.c_str(), std::fstream::in | std::fstream::binary);
        if (false == fileModel.is_open())
            return 1;

        fileModel.seekg(0, std::fstream::end);
        int dataSize = int(fileModel.tellg());
        fileModel.seekg(0, std::fstream::beg);

        AutoArray<char> encryptedData(dataSize);
        fileModel.read(encryptedData.begin(), dataSize);
        fileModel.close();

        int *pBuffer = reinterpret_cast<int *>(encryptedData.begin());
        // encrypt data by shift left		
        int numOfData = dataSize / sizeof(pBuffer[0]);
        for (int i = 0; i < numOfData; ++i)
        {
            int tempData = pBuffer[i];
            pBuffer[i] = hzx::ror(static_cast<unsigned int>(tempData),
                hzx::g_shiftBits);
        }

        const int modelnumber = pBuffer[0];
        std::vector<int> protoTxtLen, modelSize;
        for (int i = 0; i < modelnumber; ++i)
        {
            protoTxtLen.push_back(pBuffer[2 * i + 1]);
            modelSize.push_back(pBuffer[2 * i + 2]);
        }
        char *pParamBuf = encryptedData.begin() + 
            sizeof(int) * (2 * modelnumber + 1);
        pWeightBuf.resize(modelSize[0]);
        memcpy(pWeightBuf.begin(), pParamBuf + protoTxtLen[0],
            modelSize[0] * sizeof(unsigned char));

        ncnn::Net *pCaffeNet = new ncnn::Net();
        pCaffeNet->load_param_mem(pParamBuf);;
        pCaffeNet->load_model(pWeightBuf.begin());

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        float NormPoints[10] = {
            95.0f, 100.0f,
            95.0f, 100.0f,
            150.0f, 150.0f,
            110.0f, 200.0f,
            190.0f, 200.0f,
        };
        float Weights[10] = {
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f,
        };

        affineNorm.Initialize(256, 256, 1.0, 300, NormPoints, Weights, 5);

		*pHandle = reinterpret_cast<BeautyHandle>(pCaffeNet);
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

int __stdcall InitOLDFaceBeauty(const char *szParamName,
    const char *szBinName, BeautyHandle *pHandle,
    int num_threads, bool light_mode)
{
    if (pHandle == NULL)
        return -1;

    // initialize deep face network
    *pHandle = NULL;
    std::locale::global(std::locale(""));

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
            strParamPath += FACE_BEAUTY_PARAM_NAME;

        if (szBinName != 0)
            strBinPath += szBinName;
        else
            strBinPath += FACE_BEAUTY_BIN_NAME;

        ncnn::Net *pCaffeNet = new ncnn::Net();
        pCaffeNet->load_param(strParamPath.c_str());
        pCaffeNet->load_model(strBinPath.c_str());

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        float NormPoints[10] = {
            95.0f, 100.0f,
            95.0f, 100.0f,
            150.0f, 150.0f,
            110.0f, 200.0f,
            190.0f, 200.0f,
        };
        float Weights[10] = {
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f,
        };

        affineNorm.Initialize(256, 256, 1.0, 300, NormPoints, Weights, 5);

        *pHandle = reinterpret_cast<BeautyHandle>(pCaffeNet);
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

int __stdcall UninitFaceBeauty(BeautyHandle handle)
{
    ncnn::Net *pCaffeNet = reinterpret_cast<ncnn::Net *>(handle);
    pCaffeNet->clear();
	delete pCaffeNet;

	return 0;
}

//int __stdcall GetDeepFeatSize(BeautyHandle handle)
//{
//	Net<float> *pCaffeNet = reinterpret_cast<Net<float> *>(handle);
//	const vector<Blob<float>*>& result = pCaffeNet->output_blobs();
//	int len = result[0]->count() * sizeof(float);
//	return len;
//}
