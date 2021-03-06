#ifndef _FACE_DETECT_ENGINE_HPP_
#define _FACE_DETECT_ENGINE_HPP_

#ifndef _WIN32
#define __stdcall
#endif

#include "dnhpx_structure.h"

#ifdef __cplusplus
extern "C" {
#endif
    // Set Module Path
    int __stdcall DNHPXSetFaceDetectLibPath(const char* szLibPath);

    // Initialized
    int __stdcall DNHPXInitFaceDetect(DNHPXFaceDetHandle* pHandle, const char* model_name = NULL);

    // Detect max face
    int __stdcall DNHPXMaxFaceDetect(DNHPXFaceDetHandle handle, const unsigned char* image_data,
        int image_width, int image_height, DNHPXFaceRect& face_box, 
        const int min_size = 40, const int num_threads = 4);

    // Detect all face
    int __stdcall DNHPXFaceDetect(DNHPXFaceDetHandle handle, const unsigned char* image_data,
        int image_width, int image_height, std::vector<DNHPXFaceRect> &face_box, 
        const int min_size = 40, 
        const int num_threads = 4);

    // Release
    int __stdcall DNHPXUninitFaceDetect(DNHPXFaceDetHandle handle);

    /*
     *	DNHPXFaceBuffering：对人脸进行磨皮美白
     *	输入参数:
     *		input_image[in]: 输入图像
     *		face_box[in]: 输入对应图像的人脸信息
     *		output_image[out]: 输出图像
     *      param[in]: 滤波参数，可以使用默认值
     *	返回值: int (参见dnhpx_error_code.h中的error code定义)
     */
    int __stdcall DNHPXFaceBuffering(const unsigned char* input_image,
        int image_width, int image_height, std::vector<DNHPXFaceRect>& face_box,
        unsigned char* output_image, FaceBufferingParam param = FaceBufferingParam());

#ifdef __cplusplus
}
#endif

#endif
