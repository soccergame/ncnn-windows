#include "face_gender.h"
#include "FaceDetectEngine.hpp"
#include "DNHPXFaceAlignment.h"
#include "autoarray.h"
#include "TimeCount.h"
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
void readFileList(const char* basePath, vector<string>& imgFiles)
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
    CTimeCount timeCount;
    
    try
    {
        // Initialize        
        retValue = DNHPXSetFaceAlignmentLibPath(pModulePath);       
        retValue |= DNHPXInitFaceAlignment();
        //retValue |= InitDeepFeat("NNModel.dat", gpuId, &hAge);
        if (0 != retValue) {
            std::cout << "Alignment error Code: " << retValue << std::endl;
            throw retValue;
        }

        retValue = FaceDetect_setLibPath(pModulePath);
        DNHPX::FDHANDLE hDetect;
        retValue |= FaceDetect_init(&hDetect);
        if (DNHPX_OK != retValue) {
            std::cout << "Detection error Code: " << retValue << std::endl;
            DNHPXUninitFaceAlignment();
            throw retValue;
        }
            
        
        // Read Image
        //cv::Mat garyImgData = cv::imread(strImgName, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat oriImgData = cv::imread(strImgName, cv::IMREAD_COLOR);
        // Face detection
        timeCount.Start();
        DNHPX::DetectedFaceBox face_box;
        retValue = FaceDetect_maxDetect(hDetect, oriImgData, face_box);
        if (0 != retValue)
            throw retValue;

        timeCount.Stop();
        std::cout << "Detection: " << 1000 * timeCount.GetTime() << "ms" << std::endl;
        
        cv::Mat cvt_image;
        cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2GRAY);

        DNHPXFaceRect det_face;
        det_face.face.left = face_box.box[0];//dets[0].left() / 2;
        det_face.face.top = face_box.box[1];// dets[0].top() / 2;
        det_face.face.right = face_box.box[2];// dets[0].right() / 2;
        det_face.face.bottom = face_box.box[3];// dets[0].bottom() / 2;
        det_face.confidence = 100;

        DNHPXEyePointF eye_points;
        eye_points.left_eye.x = face_box.keypoints[0];
        eye_points.left_eye.y = face_box.keypoints[1];
        eye_points.right_eye.x = face_box.keypoints[2];
        eye_points.right_eye.y = face_box.keypoints[3];
        eye_points.confidence = 100;
        DNHPXPointF key_points[88];

        timeCount.Start();
        retValue = DNHPXFaceAlignmentEx(cvt_image.data, cvt_image.cols, 
            cvt_image.rows, &eye_points, key_points);
        timeCount.Stop();
        std::cout << "Alignment: " << 1000 * timeCount.GetTime() << "ms" << std::endl;
        
        if (DNHPX_OK == retValue) {
            std::string image_name = strImgName;
            std::string result_name;
            size_t found = image_name.rfind(".");
            if (found != std::string::npos)
                result_name = image_name.replace(found, string::npos, "_88.jpg");
            else
                result_name = image_name + "_88.jpg";

            cv::Mat result_img = oriImgData.clone();
            // gray_image = cv::imread(argv[i], cv::IMREAD_COLOR);
            //fp.open(result_name, fstream::out);
            for (int j = 0; j < 88; ++j) {
                cv::circle(result_img,
                    cv::Point(int(key_points[j].x + 0.5), int(key_points[j].y + 0.5)),
                    2,
                    cv::Scalar(255, 0, 0),
                    2);
            }
            cv::imwrite(result_name, result_img);
        }
        else
            cout << "Can not read images!" << endl;
        
        DNHPXUninitFaceAlignment();
        FaceDetect_release(hDetect);
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


