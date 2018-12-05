#pragma once

#ifndef _WIN32
#define __stdcall
#endif

#ifdef __cplusplus
extern "C"
{
#endif	//	__cplusplus

#ifndef _THIDFACEHANDLE
#define _THIDFACEHANDLE
	typedef void *GenderHandle;
#endif	//	_THIDFACEHANDLE

	/**
	 *	\brief set feature extraction library path
	 *		\param[in] szLibPath library path name
	 *	\return int error code defined in THIDErrorDef.h
	 */
	int __stdcall SetFaceGenderLibPath(const char *szLibPath);

	/**
	 *	\brief initialize deep face feature extraction sdk	 
	 *	\return int error code defined in THIDErrorDef.h	 
	 */
    int __stdcall InitFaceGender(const char *szResName,
        GenderHandle *pHandle, int num_threads = 1, bool light_mode = true);

    /**
    *	\brief initialize deep face feature extraction sdk
    *	\return int error code defined in THIDErrorDef.h
    */
    int __stdcall InitOLDFaceGender(const char *szParamName,
        const char *szBinName, GenderHandle *pHandle,
        int num_threads = 1, bool light_mode = true);
	

	/**
	 *	\brief free deep face feature extraction sdk
	 *	\return int error code defined in THIDErrorDef.h
	 */
	int __stdcall UninitFaceGender(GenderHandle handle);

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
    int __stdcall GetFaceGenderScore(GenderHandle handle,
        const float *feaPoints, const unsigned char *image_data,int width, 
        int height, int channel, float &gender_score, int &age, float &beauty_score);
#ifdef __cplusplus
}
#endif
