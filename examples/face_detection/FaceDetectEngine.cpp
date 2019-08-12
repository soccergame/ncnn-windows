#include "FaceDetectEngine.hpp"
#include "mtcnn.h"
#include "ErrorCodeDef.h"

#if defined( _ANDROID) || defined(_IOS) || defined(__GNUC__)
#include <cerrno>
#include <cstddef>
#define _countof(x) (sizeof(x)/sizeof(x[0]))
#define strcpy_s(x,y,z) strncpy((x), (z), (y))
#else
#include "windows.h"
#endif

#include <memory>
#include <string>

#define FACE_DETECTION_MODEL_NAME "libsnfd.so"

char g_szFaceDetectionDLLPath[_MAX_PATH] = { 0 };

namespace {
    typedef struct _face_engine
    {
        MTCNN *pFaceDetect;
    }FaceDetectEngineData;

    volatile bool g_bFaceDetectionInited = false;
    volatile int g_FaceDetectionInitCount = 0;	
}

int __stdcall FaceDetect_setLibPath(const char *model_path)
{
    if (model_path == NULL)
        return INVALID_INPUT;

    if (g_bFaceDetectionInited)
        return OK;

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

int __stdcall FaceDetect_init(SN::FDHANDLE *pHandle, const char *model_name)
{
    if (pHandle == NULL)
        return -1;

    // initialize deep face network
    *pHandle = NULL;
    std::locale::global(std::locale(""));

    if (g_bFaceDetectionInited)
    {
        ++g_FaceDetectionInitCount;
        return OK;
    }

	int res = OK;

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
        *pHandle = (SN::FDHANDLE)pEngineData;
#ifndef OLD_NCNN
        pEngineData->pFaceDetect = new MTCNN;
        res = pEngineData->pFaceDetect->Init(strDllPath);
        if (OK != res) {
            delete pEngineData;
            throw res;
        }
#else
        pEngineData->pFaceDetect = new MTCNN(g_szFaceDetectionDLLPath);
        //res = pEngineData->pFaceDetect->Init(strDllPath);
        if (!pEngineData->pFaceDetect) {
            delete pEngineData;
            throw res;
        }
#endif

        g_bFaceDetectionInited = true;
        ++g_FaceDetectionInitCount;

        *pHandle = reinterpret_cast<SN::FDHANDLE>(pEngineData);
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

int __stdcall FaceDetect_maxDetect(SN::FDHANDLE handle, 
    const cv::Mat &image, SN::DetectedFaceBox &face_box, 
    const float min_size, const int num_threads){
    
    if(NULL == handle){
        //LOG(ERROR) << "handle == NULL!" << endl;
        return INVALID_INPUT;
    }
	if (min_size < 20 || min_size>200)
	{
		return INVALID_FACE_RECT;
	}
    if (image.data == NULL) {
        return INVALID_INPUT;
    }

    if (!g_bFaceDetectionInited)
        return MODEL_NOT_INITIALIZED;

    int res = OK;

    try {
        FaceDetectEngineData *pEngineData = 
            reinterpret_cast<FaceDetectEngineData *>(handle);

        pEngineData->pFaceDetect->SetMinFace(min_size);
        pEngineData->pFaceDetect->SetNumThreads(num_threads);

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
        //mtcnn face detection
        std::vector<Bbox> finalBbox;
        pEngineData->pFaceDetect->detectMaxFace(ncnn_img, finalBbox);
        const int num_box = finalBbox.size();
        if (num_box <1)
        {
            //LOG(ERROR) << "detect no face!" << endl;
            return NO_FACE;
        }
        face_box.box[0] = (finalBbox[0].x1 < 0) ? 0 : finalBbox[0].x1;
        face_box.box[1] = (finalBbox[0].y1 < 0) ? 0 : finalBbox[0].y1;
        face_box.box[2] = (finalBbox[0].x2 > image.cols - 1) ? image.cols - 1 : finalBbox[0].x2;
        face_box.box[3] = (finalBbox[0].y2 > image.rows - 1) ? image.rows - 1 : finalBbox[0].y2;
        face_box.score = finalBbox[0].score;

        for (int i = 0; i < 5; i++)
        {
            face_box.keypoints[2 * i] = finalBbox[0].ppoint[i];
            face_box.keypoints[2 * i + 1] = finalBbox[0].ppoint[i + 5];
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

int __stdcall FaceDetect_Detect(SN::FDHANDLE handle,
    const cv::Mat &image, std::vector<SN::DetectedFaceBox> &face_box,
    const float min_size, const int num_threads) {
    if (NULL == handle) {
        //LOG(ERROR) << "handle == NULL!" << endl;
        return INVALID_INPUT;
    }
    if (min_size < 20 || min_size>200)
    {
        return INVALID_FACE_RECT;
    }
    if (image.data == NULL) {
        return INVALID_INPUT;
    }

    if (!g_bFaceDetectionInited)
        return MODEL_NOT_INITIALIZED;

    int res = OK;

    try {
        FaceDetectEngineData *pEngineData =
            reinterpret_cast<FaceDetectEngineData *>(handle);

        pEngineData->pFaceDetect->SetMinFace(min_size);
        pEngineData->pFaceDetect->SetNumThreads(num_threads);

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
        //mtcnn face detection
        std::vector<Bbox> finalBbox;
        pEngineData->pFaceDetect->detect(ncnn_img, finalBbox);
        const int num_box = finalBbox.size();
        
        if (num_box <1)
        {
            //LOG(ERROR) << "detect no face!" << endl;
            return NO_FACE;
        }
        face_box.resize(num_box);
        for (int j = 0; j < num_box; ++j) {
            face_box[j].box[0] = (finalBbox[j].x1 < 0) ? 0 : finalBbox[j].x1;
            face_box[j].box[1] = (finalBbox[j].y1 < 0) ? 0 : finalBbox[j].y1;
            face_box[j].box[2] = (finalBbox[j].x2 > image.cols - 1) ? image.cols - 1 : finalBbox[j].x2;
            face_box[j].box[3] = (finalBbox[j].y2 > image.rows - 1) ? image.rows - 1 : finalBbox[j].y2;
            face_box[j].score = finalBbox[j].score;

            for (int i = 0; i < 5; i++)
            {
                face_box[j].keypoints[2 * i] = finalBbox[j].ppoint[i];
                face_box[j].keypoints[2 * i + 1] = finalBbox[j].ppoint[i + 5];
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

int __stdcall FaceDetect_release(SN::FDHANDLE handle)
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

	return OK;
}


