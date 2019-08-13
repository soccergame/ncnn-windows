#ifndef _FACE_DETECT_ENGINE_HPP_
#define _FACE_DETECT_ENGINE_HPP_

#ifndef _WIN32
#define __stdcall
#endif

#include <opencv2/core/core.hpp>

namespace DNHPX {
#ifndef FACE_DETECTION_HANDLE
#define FACE_DETECTION_HANDLE
    typedef  void*  FDHANDLE;
#endif

#ifndef FACE_DETECTED_BOX
#define FACE_DETECTED_BOX
    struct DetectedFaceBox {
        // xmin, ymin, xmax, ymax
        float box[4]; 
        // x1,y1,x2,y2,x3,y3,x4,y4,x5,y5
        // (left_eye, right_eye, nose, left_mouth, right_mouth) 
        float keypoints[10]; 
        float score;
    };
#endif
}


#ifdef __cplusplus

extern "C" {
#endif
    // Set Module Path
    int __stdcall FaceDetect_setLibPath(const char *model_path);

    // Initialized
    int __stdcall FaceDetect_init(DNHPX::FDHANDLE *pHandle,
        const char *model_name = NULL);

    // Detect max face
    int __stdcall FaceDetect_maxDetect(DNHPX::FDHANDLE handle,
        const cv::Mat &image, DNHPX::DetectedFaceBox &face_box,
        const float min_size = 40, const int num_threads = 4);

    // Detect all face
    int __stdcall FaceDetect_Detect(DNHPX::FDHANDLE handle,
        const cv::Mat &image, std::vector<DNHPX::DetectedFaceBox> &face_box,
        const float min_size = 40, const int num_threads = 4);

    // Release
    int __stdcall FaceDetect_release(DNHPX::FDHANDLE handle);

#ifdef __cplusplus
}
#endif

#endif
