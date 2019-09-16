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
	int __stdcall DNHPXSetFaceGenderLibPath(const char *szLibPath);

	/**
	 *	\brief initialize deep face feature extraction sdk	 
	 *	\return int error code defined in THIDErrorDef.h	 
	 */
    int __stdcall DNHPXInitFaceGender(const char* szResName,
        DNHPXFaceAttHandle* pHandle, int num_threads = 1, bool light_mode = true);

    /**
    *	\brief initialize deep face feature extraction sdk
    *	\return int error code defined in THIDErrorDef.h
    */
    int __stdcall DNHPXInitOLDFaceGender(const char* szParamName,
        const char *szBinName, DNHPXFaceAttHandle* pHandle,
        int num_threads = 1, bool light_mode = true);
	

	/**
	 *	\brief free deep face feature extraction sdk
	 *	\return int error code defined in THIDErrorDef.h
	 */
	int __stdcall DNHPXUninitFaceGender(DNHPXFaceAttHandle handle);

	/**
	 *	\brief get deep face feature size in bytes
	 *	\return int face feature size in bytes
	 */
    /*int __stdcall GetDeepFeatSize(BeautyHandle handle);*/
    /**
    *	���ܣ� ��ȡ��ֵ����
    *	���룺BeautyHandle ģ��ָ��
    *         feaPoints ����ؼ���
    *         image_data raw��ʽ��ͼ�����ݣ�����RGB����
    *         width ͼ��Ŀ�
    *         height ͼ��ĸ�
    *         channel ͼ���ͨ����
    *   �����
    *         pFeatures ���ص���ֵ����
    *         fea_dim ���ص�pFeatures�ĳ���
    */
    int __stdcall DNHPXGetFaceGenderScore(DNHPXFaceAttHandle handle,
        const DNHPXPointF *feaPoints, const unsigned char *image_data,int width, 
        int height, int channel, float &gender_score, int &age, float &beauty_score,
        float &glass_score, int &emotion, float &happy_score);
#ifdef __cplusplus
}
#endif