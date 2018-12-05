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
    int __stdcall GetFaceGenderScore(GenderHandle handle,
        const float *feaPoints, const unsigned char *image_data,int width, 
        int height, int channel, float &gender_score, int &age, float &beauty_score);
#ifdef __cplusplus
}
#endif