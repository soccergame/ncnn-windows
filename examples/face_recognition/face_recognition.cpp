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

#include "face_recognition.h"
#include "NormFaceImage.h"
#include "autoarray.h"
#include "Error_Code.h"
#include "net.h"


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
    hzx::CNormImageSimilarity affineNorm;
    AutoArray<unsigned char> pWeightBuf;
    const float mean_vals[3] = { 127.5, 127.5, 127.5 };
    const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };

    volatile bool g_bFaceRecognitionInited = false;
    volatile int g_FaceRecognitionInitCount = 0;
}

int __stdcall GetFaceRecognitionFeature(RecognitionHandle handle,
    const float *feaPoints, const unsigned char *image_data, int width,
    int height, int channel, float **feature, int &fea_dim)
{
    if (image_data == 0 || width <= 0 || height <= 0)
        return  INVALID_IMAGE;

    if (1 != channel && 3 != channel)
        return INVALID_IMAGE_FORMAT;

    if (!g_bFaceRecognitionInited)
        return MODEL_NOT_INITIALIZED;

    int nRet = 0;
    //clock_t count;
    try
    {
        ncnn::Mat ncnn_face_img(112, 112, 3, 4u);
        if (ncnn_face_img.empty())
            return INVALID_IMAGE;

        //count = clock();
        int size = width * height;
        if (channel == 3) {
            affineNorm.NormImageRaw2Planar(image_data, width, height,
                channel, feaPoints, 5, (float *)ncnn_face_img);
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

        ncnn::Net *pCaffeNet = reinterpret_cast<ncnn::Net *>(handle);
        ncnn::Extractor ex = pCaffeNet->create_extractor();
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
    RecognitionHandle *pHandle, int num_threads, bool light_mode)
{
	if (pHandle == NULL)
		return -1;
	
	// initialize deep face network
	*pHandle = NULL;
	std::locale::global(std::locale(""));

    if (g_bFaceRecognitionInited)
    {
        ++g_FaceRecognitionInitCount;
        return OK;
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

        std::fstream fileModel;
        fileModel.open(strDllPath.c_str(), std::fstream::in | std::fstream::binary);
        if (false == fileModel.is_open())
            return 1;

        fileModel.seekg(0, std::fstream::end);
        int dataSize = int(fileModel.tellg());
        fileModel.seekg(0, std::fstream::beg);

        //CMyFile fileModel(strDllPath.c_str(), CMyFile::modeRead);
        //int dataSize = static_cast<int>(fileModel.GetLength());
        AutoArray<char> encryptedData(dataSize);
        //fileModel.Read(encryptedData, dataSize);
        fileModel.read(encryptedData.begin(), dataSize);
        //fileModel.Close();
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

        /*float NormPoints_128[10] = {
            35.5f, 55.48f,
            91.5f, 55.48f,
            63.5f, 83.98f,
            45.5f, 111.48f,
            81.5f, 111.48f,
        };
        affineNorm.Initialize(96, 128, 0.78125, 128, NormPoints_128);*/
        affineNorm.Initialize(112, 112, 1.0);

		*pHandle = reinterpret_cast<RecognitionHandle>(pCaffeNet);

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
    const char *szBinName, RecognitionHandle *pHandle,
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
        return OK;
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

        ncnn::Net *pCaffeNet = new ncnn::Net();
        pCaffeNet->load_param(strParamPath.c_str());
        pCaffeNet->load_model(strBinPath.c_str());

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        /*float NormPoints_128[10] = {
            35.5f, 55.48f,
            91.5f, 55.48f,
            63.5f, 83.98f,
            45.5f, 111.48f,
            81.5f, 111.48f,
        };*/
        affineNorm.Initialize(112, 112, 1.0);

        *pHandle = reinterpret_cast<RecognitionHandle>(pCaffeNet);

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

int __stdcall UninitFaceRecognition(RecognitionHandle handle)
{
    if (g_bFaceRecognitionInited) {
        --g_FaceRecognitionInitCount;
        if (g_FaceRecognitionInitCount == 0) {

            // 初始化变量修改
            g_bFaceRecognitionInited = false;

            ncnn::Net *pCaffeNet = reinterpret_cast<ncnn::Net *>(handle);
            pCaffeNet->clear();
            delete pCaffeNet;
        }
    }
	return 0;
}
