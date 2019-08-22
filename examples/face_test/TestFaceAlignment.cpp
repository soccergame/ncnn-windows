//#include "face_gender.h"
#include "dnhpx_face_detection.h"
#include "dnhpx_face_alignment.h"
//#include "dnhpx_auto_array.h"
#include "dnhpx_time_count.h"
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <assert.h>
#include <opencv2/opencv.hpp>

#ifndef _WIN32
#include<unistd.h> 
#include <dirent.h>
#endif

#ifndef _WIN32
void readFileList(const char* basePath, std::vector<std::string>& imgFiles)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir = opendir(basePath)) == NULL)
    {
        return;
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 ||
            strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 8)//file 
        {
            int len = strlen(ptr->d_name);
            // jpg, jpeg, png, bmp
            if ((ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'p' && ptr->d_name[len - 3] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'e' && ptr->d_name[len - 3] == 'p' && ptr->d_name[len - 4] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'n' && ptr->d_name[len - 3] == 'p') || (ptr->d_name[len - 1] == 'p' && ptr->d_name[len - 2] == 'm' && ptr->d_name[len - 3] == 'b'))
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if (ptr->d_type == 10)/// link file
        {
            int len = strlen(ptr->d_name);
            if ((ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'p' && ptr->d_name[len - 3] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'e' && ptr->d_name[len - 3] == 'p' && ptr->d_name[len - 4] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'n' && ptr->d_name[len - 3] == 'p') || (ptr->d_name[len - 1] == 'p' && ptr->d_name[len - 2] == 'm' && ptr->d_name[len - 3] == 'b'))
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if (ptr->d_type == 4)//dir
        {
            memset(base, '\0', sizeof(base));
            strcpy(base, basePath);
            strcat(base, "/");
            strcat(base, ptr->d_name);
            readFileList(base, imgFiles);
        }
    }
    closedir(dir);
}
#endif

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "usage:" << std::endl;
        std::cout << "mtcnn_test <module_path> <filename>" << std::endl;
        return -1;
    }
    std::string modulePath = argv[1];
    const char* pModulePath = modulePath.c_str();
        
    std::string strImgName = argv[2];
    int retValue = 0;
    float maxR = 0;
    int label = 0;
    dnhpx::CTimeCount timeCount;
    
    try
    {
        // Initialize        
        retValue = DNHPXSetFaceAlignmentLibPath(pModulePath);  
        DNHPXFaceAliHandle align_handle;
        retValue |= DNHPXInitFaceAlignment(&align_handle);
        //retValue |= InitDeepFeat("NNModel.dat", gpuId, &hAge);
        if (0 != retValue) {
            std::cout << "Alignment error Code: " << retValue << std::endl;
            throw retValue;
        }

        retValue = DNHPXSetFaceDetectLibPath(pModulePath);
        DNHPXFaceDetHandle hDetect;
        retValue |= DNHPXInitFaceDetect(&hDetect);
        if (DNHPX_OK != retValue) {
            std::cout << "Detection error Code: " << retValue << std::endl;
            DNHPXUninitFaceAlignment(align_handle);
            throw retValue;
        }
            
        
        // Read Image
        //cv::Mat garyImgData = cv::imread(strImgName, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat oriImgData = cv::imread(strImgName, cv::IMREAD_COLOR);

        // Face detection
        timeCount.Start();
        DNHPXFaceRect face_box;
        retValue = DNHPXMaxFaceDetect(hDetect, oriImgData.data, oriImgData.cols, 
            oriImgData.rows, face_box);
        if (0 != retValue)
            throw retValue;

        timeCount.Stop();
        std::cout << "Detection: " << 1000 * timeCount.GetTime() << "ms" << std::endl;

        /*cv::Mat buffering_image;
        DNHPXFaceBuffering(oriImgData, std::vector<DNHPXFaceRect>(1, face_box), buffering_image);*/
        std::string image_name = strImgName;
        std::string result_name;
        size_t found = image_name.rfind(".");
        /*if (found != std::string::npos)
            result_name = image_name.replace(found, std::string::npos, "_buffering.jpg");
        else
            result_name = image_name + "_buffering.jpg";

        cv::imwrite(result_name, buffering_image);*/
        
        cv::Mat cvt_image;
        cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2GRAY);

        DNHPXEyePointF eye_points;
        eye_points.left_eye = face_box.key_points[0];
        eye_points.right_eye = face_box.key_points[1];
        // 如果输入眼睛位置（或者有可能是五个关键点位置），置信度一定要设为大于0
        eye_points.confidence = 100.f;
        DNHPXPointF key_points[FEATURE_POINT_NUM];

        timeCount.Start();
#ifdef USE_ALIGNMENT_OLD
        retValue = DNHPXFaceAlignmentEx(align_handle, cvt_image.data, cvt_image.cols,
            cvt_image.rows, &eye_points, key_points);
#else
        retValue = DNHPXAlignmentFromFace(align_handle, oriImgData.data, oriImgData.cols,
            oriImgData.rows, &face_box, key_points);
#endif
        timeCount.Stop();
        std::cout << "Alignment: " << 1000 * timeCount.GetTime() << "ms" << std::endl;
        
        if (DNHPX_OK == retValue) {
            image_name = strImgName;
            if (found != std::string::npos)
                result_name = image_name.replace(found, std::string::npos, "_88.jpg");
            else
                result_name = image_name + "_88.jpg";

            cv::Mat result_img = oriImgData.clone();
            // gray_image = cv::imread(argv[i], cv::IMREAD_COLOR);
            //fp.open(result_name, fstream::out);
            for (int j = 0; j < FEATURE_POINT_NUM; ++j) {
                cv::circle(result_img,
                    cv::Point(int(key_points[j].x + 0.5), int(key_points[j].y + 0.5)),
                    2,
                    cv::Scalar(255, 0, 0),
                    2);
            }
            cv::rectangle(result_img, cv::Rect(face_box.face.left, face_box.face.top,
                face_box.face.width(), face_box.face.height()), cv::Scalar(255, 0, 0),
                2);
            cv::imwrite(result_name, result_img);
        }
        else
            std::cout << "Can not read images!" << std::endl;
        
        DNHPXUninitFaceAlignment(align_handle);
        DNHPXUninitFaceDetect(hDetect);
    }
    catch (const std::bad_alloc &)
	{
		retValue = -2;
	}
	catch (const int &errCode)
	{
		retValue = errCode;
	}
	catch (...)
	{
		retValue = -3;
	}
    
	return retValue;
}


