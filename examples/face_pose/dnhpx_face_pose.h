#pragma once

#ifndef _WIN32
#define __stdcall
#endif

#include "dnhpx_utility.h"

#ifdef __cplusplus
extern "C"
{
#endif	//	__cplusplus

    /**
     *	\brief set feature extraction library path
     *		\param[in] szLibPath library path name
     *	\return int error code defined in THIDErrorDef.h
     */
    int __stdcall DNHPXSetFacePoseLibPath(const char* szLibPath);

    /**
     *	\brief initialize deep face feature extraction sdk
     *	\return int error code defined in THIDErrorDef.h
     */
    int __stdcall DNHPXInitFacePose(const char* szResName,
        DNHPXFacePosHandle* pHandle, int num_threads = 1, bool light_mode = true);

    /**
    *	\brief initialize deep face feature extraction sdk
    *	\return int error code defined in THIDErrorDef.h
    */
    int __stdcall DNHPXInitOLDFacePose(const char* szParamName,
        const char* szBinName, DNHPXFacePosHandle* pHandle,
        int num_threads = 1, bool light_mode = true);


    /**
     *	\brief free deep face feature extraction sdk
     *	\return int error code defined in THIDErrorDef.h
     */
    int __stdcall DNHPXUninitFacePose(DNHPXFacePosHandle handle);

     /**
     *	功能： 获取人脸姿态角度
     *	输入：DNHPXFacePosHandle 模型指针
     *        faceRect 人脸框
     *        pbyGrayImage raw格式的图像数据，灰度图
     *        width 图像的宽
     *        height 图像的高
     *   输出：
     *         pitch 返回的俯仰角
     *         yaw 返回的姿态角（摇摆角）
     *         roll 返回的旋转角
     */
    int __stdcall DNHPXGetFacePose(DNHPXFacePosHandle handle,
        const DNHPXFaceRect& faceRect, const unsigned char* pbyGrayImage, int width,
        int height, float& pitch, float& yaw, float& roll);
#ifdef __cplusplus
}
#endif
