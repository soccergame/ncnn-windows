/*
# 作者：杨睿昕、何智翔
# 描述：建立网络一些自定层
# 创建日期：20191112
*/

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

#include "dnhpx_face_pose.h"
#include "dnhpx_face_normalization.h"
#include "dnhpx_auto_array.h"
#include "dnhpx_error_code.h"

#ifndef _WIN32
#define _MAX_PATH 260
#endif

#define FACE_POSE_MODEL_NAME "librlz.so"
#define FACE_POSE_PARAM_NAME "librlz.param"
#define FACE_POSE_BIN_NAME "librlz.bin"

namespace
{
    char g_szDeepFeatSDKPath[_MAX_PATH] = { 0 };
    int g_num_threads = 1;
    bool g_light_mode = true;
    const float mean_vals[1] = { 127.5f};
    const float norm_vals[1] = { 0.0078125f};

    volatile bool g_bFacePoseInited = false;
    volatile int g_FacePoseInitCount = 0;

    double postProcessing(std::vector<double> pred) {
        std::vector<double>::iterator max = std::max_element(std::begin(pred), std::end(pred));
        double maxVal = *max;
        // std::cout << "maxVal " << maxVal << std::endl;
        std::vector<double> softmaxPred;
        double sumTmp = 0.0;
        for (int i = 0; i < 66; ++i)
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
}

int __stdcall DNHPXSetFacePoseLibPath(const char* szLibPath)
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

int __stdcall DNHPXInitFacePose(const char* szNetName,
    DNHPXFaceAttHandle* pHandle, int num_threads, bool light_mode)
{
    if (pHandle == NULL)
        return -1;

    // initialize deep face network
    *pHandle = NULL;
    std::locale::global(std::locale(""));

    if (g_bFacePoseInited)
    {
        ++g_FacePoseInitCount;
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
            strDllPath += FACE_POSE_MODEL_NAME;

        dnhpx::CAlgorithmDomain* pCaffeNet = new dnhpx::CAlgorithmDomain();
        pCaffeNet->init(strDllPath.c_str());

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        *pHandle = reinterpret_cast<DNHPXFaceAttHandle>(pCaffeNet);

        g_bFacePoseInited = true;
        ++g_FacePoseInitCount;
#endif
    }
    catch (const std::bad_alloc&)
    {
        retValue = -2;
    }
    catch (const int& errCode)
    {
        retValue = errCode;
    }
    catch (...)
    {
        retValue = -3;
    }

    return retValue;
}

int __stdcall DNHPXInitOLDFacePose(const char* szParamName,
    const char* szBinName, DNHPXFaceAttHandle* pHandle,
    int num_threads, bool light_mode)
{
    printf("该函数即将停止使用，请使用函数：DNHPXInitFacePose\n");

    if (pHandle == NULL)
        return -1;

    // initialize deep face network
    *pHandle = NULL;
    std::locale::global(std::locale(""));

    if (g_bFacePoseInited)
    {
        ++g_FacePoseInitCount;
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
            strParamPath += FACE_POSE_PARAM_NAME;

        if (szBinName != 0)
            strBinPath += szBinName;
        else
            strBinPath += FACE_POSE_BIN_NAME;

        dnhpx::CAlgorithmDomain* pCaffeNet = new dnhpx::CAlgorithmDomain();
        pCaffeNet->init(std::vector<std::string>(1, strParamPath),
            std::vector<std::string>(1, strBinPath));

        g_num_threads = num_threads;
        g_light_mode = light_mode;

        *pHandle = reinterpret_cast<DNHPXFaceAttHandle>(pCaffeNet);

        g_bFacePoseInited = true;
        ++g_FacePoseInitCount;
    }
    catch (const std::bad_alloc&)
    {
        retValue = -2;
    }
    catch (const int& errCode)
    {
        retValue = errCode;
    }
    catch (...)
    {
        retValue = -3;
    }

    return retValue;
}

int __stdcall DNHPXUninitFacePose(DNHPXFaceAttHandle handle)
{
    if (g_bFacePoseInited) {
        --g_FacePoseInitCount;
        if (g_FacePoseInitCount == 0) {

            // 如果满足被释放的条件，则将初始化参数设置为false
            g_bFacePoseInited = false;

            dnhpx::CAlgorithmDomain* pCaffeNet = reinterpret_cast<dnhpx::CAlgorithmDomain*>(handle);
            pCaffeNet->clear();
            delete pCaffeNet;
        }
    }
    return 0;
}

int __stdcall DNHPXGetFacePose(DNHPXFacePosHandle handle,
    const DNHPXFaceRect& faceRect, const unsigned char* pbyGrayImage, int width,
    int height, float& pitch, float& yaw, float& roll)
{
    if (pbyGrayImage == 0 || width <= 0 || height <= 0)
        return  DNHPX_INVALID_INPUT;

    if (!g_bFacePoseInited)
        return DNHPX_MODEL_NOT_INITIALIZED;

    int nRet = 0;

    try
    {
        ncnn::Mat src = ncnn::Mat::from_pixels(pbyGrayImage, ncnn::Mat::PIXEL_GRAY,
            width, height);

        ncnn::Mat dst;
        ncnn::copy_cut_border(src, dst, faceRect.face.top, height - faceRect.face.bottom,
            faceRect.face.left, width - faceRect.face.right);

        ncnn::Mat ncnn_face_img;
        ncnn::resize_bilinear(dst, ncnn_face_img, 48, 48);
        if (ncnn_face_img.empty())
            return DNHPX_MEMORY_ALLOC_ERROR;

        dnhpx::CAlgorithmDomain* pCaffeNet = reinterpret_cast<dnhpx::CAlgorithmDomain*>(handle);
        ncnn::Extractor ex = pCaffeNet->get_model()->create_extractor();
        ex.set_light_mode(g_light_mode);
        ex.set_num_threads(g_num_threads);

        ex.input("data", ncnn_face_img);
        ncnn::Mat out;
        // 最终的输出层
        ex.extract("hybridsequential0_multitask0_dense0_fwd", out); 

    // post processing	
        std::vector<double> predPitch, predRoll, predYaw;
        for (int i = 0; i < 66; ++i) {
            predPitch.push_back(out[i]);
            // std::cout << out[i] << std::endl;
        }
        for (int i = 66; i < 132; ++i) {
            predRoll.push_back(out[i]);
        }
        for (int i = 132; i < 198; ++i) {
            predYaw.push_back(out[i]);
        }

        // pitch times -1 to keep consistence with other api
        pitch = float(-1.0 * postProcessing(predPitch));
        roll = float(postProcessing(predRoll));
        yaw = float(postProcessing(predYaw));
    }
    catch (const std::bad_alloc&)
    {
        nRet = -2;
    }
    catch (const int& errCode)
    {
        nRet = errCode;
    }
    catch (...)
    {
        nRet = -3;
    }

    return nRet;
    
}