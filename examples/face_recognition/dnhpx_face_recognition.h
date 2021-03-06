#pragma once

#ifndef _WIN32
#define __stdcall
#endif

#include "dnhpx_structure.h"

#ifdef __cplusplus
extern "C"
{
#endif	//	__cplusplus


	/**
	 *	\brief set feature extraction library path
	 *		\param[in] szLibPath library path name
	 *	\return int error code defined in THIDErrorDef.h
	 */
	int __stdcall SetFaceRecognitionLibPath(const char *szLibPath);

	/**
	 *	\brief initialize deep face feature extraction sdk	 
	 *	\return int error code defined in THIDErrorDef.h	 
	 */
    int __stdcall InitFaceRecognition(const char *szResName,
        DNHPXFaceRecogHandle*pHandle, int num_threads = 1, bool light_mode = true);

    /**
    *	\brief initialize deep face feature extraction sdk
    *	\return int error code defined in THIDErrorDef.h
    */
    int __stdcall InitOLDFaceRecognition(const char *szParamName,
        const char *szBinName, DNHPXFaceRecogHandle*pHandle,
        int num_threads = 1, bool light_mode = true);
	

	/**
	 *	\brief free deep face feature extraction sdk
	 *	\return int error code defined in THIDErrorDef.h
	 */
	int __stdcall UninitFaceRecognition(DNHPXFaceRecogHandle handle);

	/**
	 *	\brief get deep face feature size in bytes
	 *	\return int face feature size in bytes
	 */
    /*int __stdcall GetDeepFeatSize(BeautyHandle handle);*/
    /**
    *	功能： 获取颜值分数
    *	输入：BeautyHandle 模型指针
    *         feaPoints 五个关键点
    *         image_data raw格式的图像数据，按照RGB排列
    *         width 图像的宽
    *         height 图像的高
    *         channel 图像的通道数
    *   输出：
    *         pFeatures 返回的颜值分数
    *         fea_dim 返回的pFeatures的长度
    */
    int __stdcall GetFaceRecognitionFeature(DNHPXFaceRecogHandle handle,
        const float *feaPoints, const unsigned char *image_data, int width, 
        int height, int channel, float **feature, int &fea_dim);

    /**
    *	功能： 获取人脸特征
    *	输入：BeautyHandle 模型指针
    *         feaPoints 五个关键点
    *         norm_data 归一化好的人脸图像，可以直接将cv::mat中的image.data传入
    *   输出：
    *         pFeatures 返回的特征
    *         fea_dim 返回的pFeatures的长度
    */
    int __stdcall GetFaceRecognitionFeatureRaw(DNHPXFaceRecogHandle handle,
        const unsigned char *norm_data, float **feature, int &fea_dim);
#ifdef __cplusplus
}
#endif
