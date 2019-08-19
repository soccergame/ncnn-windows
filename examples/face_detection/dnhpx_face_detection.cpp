#include "dnhpx_face_detection.h"
#include "mtcnn.h"
#include "dnhpx_error_code.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#if defined( _ANDROID) || defined(_IOS) || defined(__GNUC__)
#include <cerrno>
#include <cstddef>
#define _countof(x) (sizeof(x)/sizeof(x[0]))
#define strcpy_s(x,y,z) strncpy((x), (z), (y))
#define _MAX_PATH 260
#else
#include "windows.h"
#endif

#include <memory>
#include <string>

#define FACE_DETECTION_MODEL_NAME "libsnfd.so"

namespace {
    typedef struct _face_engine
    {
        mtcnn::CFaceDetection *pFaceDetect;
    }FaceDetectEngineData;

    char g_szFaceDetectionDLLPath[_MAX_PATH] = { 0 };
    volatile bool g_bFaceDetectionInited = false;
    volatile int g_FaceDetectionInitCount = 0;	
}

int __stdcall DNHPXSetFaceDetectLibPath(const char *model_path)
{
    if (model_path == NULL)
        return DNHPX_INVALID_INPUT;

    if (g_bFaceDetectionInited)
        return DNHPX_OK;

#ifdef _WIN32
    strcpy_s(g_szFaceDetectionDLLPath, _MAX_PATH, model_path);
#else
    strncpy(g_szFaceDetectionDLLPath, model_path, _MAX_PATH);
#endif

    size_t len = strlen(g_szFaceDetectionDLLPath);
    if (len != 0)
    {
#ifdef _WIN32
        if (g_szFaceDetectionDLLPath[len - 1] != '\\')
            strcat_s(g_szFaceDetectionDLLPath, "\\");
#else
        if (g_szFaceDetectionDLLPath[len - 1] != '/')
            strncat(g_szFaceDetectionDLLPath, "/", _MAX_PATH);
#endif
    }

    return 0;
}

int __stdcall DNHPXInitFaceDetect(DNHPXFaceDetHandle* pHandle, const char *model_name)
{
    if (pHandle == NULL)
        return -1;

    // initialize deep face network
    *pHandle = NULL;
    std::locale::global(std::locale(""));

    if (g_bFaceDetectionInited)
    {
        ++g_FaceDetectionInitCount;
        return DNHPX_OK;
    }

	int res = DNHPX_OK;

#ifndef _WIN32	
    if (strlen(g_szFaceDetectionDLLPath) == 0)
        strncpy(g_szFaceDetectionDLLPath, "./", _MAX_PATH);
#endif

    try {

        std::string strDllPath;
        strDllPath = g_szFaceDetectionDLLPath;
        if (NULL != model_name)
            strDllPath += model_name;
        else
            strDllPath += FACE_DETECTION_MODEL_NAME;

        FaceDetectEngineData *pEngineData = new FaceDetectEngineData;
        *pHandle = (DNHPXFaceDetHandle)pEngineData;
#ifndef OLD_NCNN
        pEngineData->pFaceDetect = new MTCNN;
        res = pEngineData->pFaceDetect->Init(strDllPath);
        if (DNHPX_OK != res) {
            delete pEngineData;
            throw res;
        }
#else
        pEngineData->pFaceDetect = new mtcnn::CFaceDetection(g_szFaceDetectionDLLPath);
        //res = pEngineData->pFaceDetect->Init(strDllPath);
        if (!pEngineData->pFaceDetect) {
            delete pEngineData;
            throw res;
        }
#endif

        g_bFaceDetectionInited = true;
        ++g_FaceDetectionInitCount;

        *pHandle = reinterpret_cast<DNHPXFaceDetHandle>(pEngineData);
    }
    catch (const std::bad_alloc &)
    {
        res = -2;
    }
    catch (const int &errCode)
    {
        res = errCode;
    }
    catch (...)
    {
        res = -3;
    }

    return res;
}

int __stdcall DNHPXMaxFaceDetect(DNHPXFaceDetHandle handle, const cv::Mat &image,
    DNHPXFaceRect& face_box, const int min_size, const int num_threads){
    
    if(NULL == handle){
        //LOG(ERROR) << "handle == NULL!" << endl;
        return DNHPX_INVALID_INPUT;
    }
	if (min_size < 20 || min_size>200)
	{
		return DNHPX_INVALID_FACE_RECT;
	}
    if (image.data == NULL) {
        return DNHPX_INVALID_INPUT;
    }

    if (!g_bFaceDetectionInited)
        return DNHPX_MODEL_NOT_INITIALIZED;

    int res = DNHPX_OK;

    try {
        FaceDetectEngineData *pEngineData = 
            reinterpret_cast<FaceDetectEngineData *>(handle);

        pEngineData->pFaceDetect->SetMinFace(min_size);
        pEngineData->pFaceDetect->SetNumThreads(num_threads);

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
        //mtcnn face detection
        std::vector<mtcnn::Bbox> finalBbox;
        pEngineData->pFaceDetect->detectMaxFace(ncnn_img, finalBbox);
        const int num_box = int(finalBbox.size());
        if (num_box <1)
        {
            //LOG(ERROR) << "detect no face!" << endl;
            return DNHPX_NO_FACE;
        }
        face_box.face.left = (finalBbox[0].x1 < 0) ? 0 : finalBbox[0].x1;
        face_box.face.top = (finalBbox[0].y1 < 0) ? 0 : finalBbox[0].y1;
        face_box.face.right = (finalBbox[0].x2 > image.cols - 1) ? image.cols - 1 : finalBbox[0].x2;
        face_box.face.bottom = (finalBbox[0].y2 > image.rows - 1) ? image.rows - 1 : finalBbox[0].y2;
        face_box.confidence = finalBbox[0].score;

        for (int i = 0; i < 5; i++)
        {
            face_box.key_points[i].x = finalBbox[0].ppoint[i];
            face_box.key_points[i].y = finalBbox[0].ppoint[i + 5];
        }
    }
    catch (const std::bad_alloc &)
    {
        res = -2;
    }
    catch (const int &errCode)
    {
        res = errCode;
    }
    catch (...)
    {
        res = -3;
    }

    return res;
}

int __stdcall DNHPXFaceDetect(DNHPXFaceDetHandle handle, const cv::Mat& image,
    std::vector<DNHPXFaceRect>& face_box, const int min_size, const int num_threads) {
    if (NULL == handle) {
        //LOG(ERROR) << "handle == NULL!" << endl;
        return DNHPX_INVALID_INPUT;
    }
    if (min_size < 20 || min_size>200)
    {
        return DNHPX_INVALID_FACE_RECT;
    }
    if (image.data == NULL) {
        return DNHPX_INVALID_INPUT;
    }

    if (!g_bFaceDetectionInited)
        return DNHPX_MODEL_NOT_INITIALIZED;

    int res = DNHPX_OK;

    try {
        FaceDetectEngineData *pEngineData =
            reinterpret_cast<FaceDetectEngineData *>(handle);

        pEngineData->pFaceDetect->SetMinFace(min_size);
        pEngineData->pFaceDetect->SetNumThreads(num_threads);

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
        //mtcnn face detection
        std::vector<mtcnn::Bbox> finalBbox;
        pEngineData->pFaceDetect->detect(ncnn_img, finalBbox);
        const int num_box = finalBbox.size();
        
        if (num_box <1)
        {
            //LOG(ERROR) << "detect no face!" << endl;
            return DNHPX_NO_FACE;
        }
        face_box.resize(num_box);
        for (int j = 0; j < num_box; ++j) {
            face_box[j].face.left = (finalBbox[j].x1 < 0) ? 0 : finalBbox[j].x1;
            face_box[j].face.top = (finalBbox[j].y1 < 0) ? 0 : finalBbox[j].y1;
            face_box[j].face.right = (finalBbox[j].x2 > image.cols - 1) ? image.cols - 1 : finalBbox[j].x2;
            face_box[j].face.bottom = (finalBbox[j].y2 > image.rows - 1) ? image.rows - 1 : finalBbox[j].y2;
            face_box[j].confidence = finalBbox[j].score;

            for (int i = 0; i < 5; i++)
            {
                face_box[j].key_points[i].x = finalBbox[j].ppoint[i];
                face_box[j].key_points[i].y = finalBbox[j].ppoint[i + 5];
            }
        }
        
    }
    catch (const std::bad_alloc &)
    {
        res = -2;
    }
    catch (const int &errCode)
    {
        res = errCode;
    }
    catch (...)
    {
        res = -3;
    }

    return res;
}

int __stdcall DNHPXUninitFaceDetect(DNHPXFaceDetHandle handle)
{
    --g_FaceDetectionInitCount;
    if (g_FaceDetectionInitCount == 0)
    {
        // 初始化变量修改
        g_bFaceDetectionInited = false;

        // 释放资源
        FaceDetectEngineData *pEngineData = 
            reinterpret_cast<FaceDetectEngineData *>(handle);
        if (NULL != pEngineData)
        {
            if (pEngineData->pFaceDetect != NULL)
                delete pEngineData->pFaceDetect;
            delete pEngineData;
        }
    }

	return DNHPX_OK;
}

// 版本1.0，暂时不考虑人脸的位置，而是对全图进行滤波
int __stdcall DNHPXFaceBuffering(const cv::Mat& input_image,
    std::vector<DNHPXFaceRect>& face_box, cv::Mat& output_image, 
    dnhpx::FaceBufferingParam param)
{
    if (input_image.data == NULL) {
        return DNHPX_INVALID_INPUT;
    }

    if (0 == face_box.size()) {
        return DNHPX_NO_FACE;
    }

    int res = DNHPX_OK;

    try {
        if (true == param.use_filter_only) {
            cv::bilateralFilter(input_image, output_image, param.radius, param.sigma_color,
                param.sigma_space);
        }
        else {
            // 1、进行双边滤波（也可以使用表面滤波，但是opencv没有现成的函数），
            //    我自己定义的函数速度上可能存在一些问题，还需要改进，准备2.0放入
            cv::Mat high_pass;
            cv::bilateralFilter(input_image, high_pass, param.radius, param.sigma_color,
                param.sigma_space);
            //cv::imwrite("D:/project/ncnn-windows/Build/x64/Debug/result.jpg", high_pass);

            // 2、提取细节边缘
            high_pass = high_pass - input_image + param.white;
            //cv::imwrite("D:/project/ncnn-windows/Build/x64/Debug/result1.jpg", high_pass);
            /*high_pass = high_pass + output_image.setTo(cv::Scalar_<unsigned char>());
            cv::imwrite("D:/project/ncnn-windows/Build/x64/Debug/result2.jpg", high_pass);*/

            // 3、高斯滤波，消除噪点
            cv::GaussianBlur(high_pass, high_pass,
                cv::Size(param.gaussian_ksize, param.gaussian_ksize),
                0);

            // 4、将图层进行叠加
            cv::Mat input_float_image, high_pass_float;
            //std::cout << high_pass.channels() << std::endl;
            high_pass.convertTo(high_pass_float, CV_32F);
            input_image.convertTo(input_float_image, CV_32F);
            //std::cout << high_pass_float.channels() << " " << input_float_image.channels() << std::endl;
            input_float_image = (input_float_image * (1.0f - param.opacity) +
                (input_float_image + 2.0f * high_pass_float - 256.0f) * param.opacity);

            //
            input_float_image.convertTo(output_image, CV_8U);
        }
        
    }
    catch (const std::bad_alloc&)
    {
        res = DNHPX_MEMORY_ALLOC_ERROR;
    }
    catch (const int& errCode)
    {
        res = errCode;
    }
    catch (...)
    {
        res = DNHPX_GENERIC_ERROR;
    }

    return res;
}
