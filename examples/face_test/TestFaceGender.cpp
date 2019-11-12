#include "dnhpx_face_gender.h"
#include "dnhpx_face_detection.h"
#include "dnhpx_auto_array.h"
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
    dnhpx::CTimeCount timeCount;
    
    try
    {
        // Initialize        
        retValue = DNHPXSetFaceGenderLibPath(pModulePath);
        
        DNHPXFaceAttHandle hFace;
        retValue |= DNHPXInitFaceGender(0, &hFace, 4);
        //retValue |= InitDeepFeat("NNModel.dat", gpuId, &hAge);
        if (0 != retValue) {
            std::cout << "Error Code: " << retValue << std::endl;
            throw retValue;
        }

        retValue = DNHPXSetFaceDetectLibPath(pModulePath);
        DNHPXFaceDetHandle hDetect;
        retValue |= DNHPXInitFaceDetect(&hDetect);
        if (DNHPX_OK != retValue) {
            std::cout << "Detection error Code: " << retValue << std::endl;
            DNHPXUninitFaceGender(hFace);
#ifndef NO_EXCEPTIONS  
            throw retValue;
#endif
        }
            
        
        // Read Image
        //cv::Mat garyImgData = cv::imread(strImgName, CV_LOAD_IMAGE_GRAYSCALE);
//#ifdef _WIN32
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
        
        /*cv::Mat cvt_image;
        cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2RGB);*/

        int age = 0;
        float gender_score = 0.0f;
        float beauty_score = 0.0f;
        float glass_score = 0.0f;
        float happy_score = 0.0f;
        float pitch = 0.0f;
        float yaw = 0.0f;
        float roll = 0.0f;
        int emotion = 0;
        //for (int test_idx = 0; test_idx < 1; ++test_idx) {
        timeCount.Start();
        retValue = DNHPXGetFaceGenderScore(hFace, face_box.key_points,
            oriImgData.data, oriImgData.cols, oriImgData.rows,
            oriImgData.channels(), gender_score, age, beauty_score,
            glass_score, emotion, happy_score);
        timeCount.Stop();
        std::cout << "Attribute: " << 1000 * timeCount.GetTime() << "ms" << std::endl;
        //}

        // ???????
        if (gender_score > 0.5f)
            std::cout << "Gender: female" << std::endl;
        else
            std::cout << "Gender: male" << std::endl;

        // ????????
        if (age <= 16)
            std::cout << "child" << std::endl;
        else if (age >= 62)
            std::cout << "old" << std::endl;
        else
            std::cout << "Age: " << age << std::endl;

        std::cout << "Beauty: " << beauty_score << std::endl;

        if (glass_score > 0.5f)
            std::cout << "No glasses" << std::endl;
        else
            std::cout << "Wear glasses" << std::endl;

        if (0 == emotion)
            std::cout << "angery" << std::endl;
        else if (1 == emotion)
            std::cout << "disgusted" << std::endl;
        else if (2 == emotion)
            std::cout << "fearful" << std::endl;
        else if (3 == emotion) 
            std::cout << "happy, score: " << happy_score << std::endl;
        else if (4 == emotion)
            std::cout << "neutral" << std::endl;
        else if (5 == emotion)
            std::cout << "sad" << std::endl;
        else if (6 == emotion)
            std::cout << "surprise" << std::endl;
//#else
//        char path_list[512] = { 0 };
//        strcpy(path_list, strImgName.c_str());
//
//        vector<string> imgList;
//        readFileList(path_list, imgList);
//
//        for (int l = 0; l < imgList.size(); l++)
//        {
//            cv::Mat oriImgData = cv::imread(imgList[l], CV_LOAD_IMAGE_COLOR);
//            // Face detection
//            DNHPXFaceRect face_box;
//            retValue = DNHPXMaxFaceDetect(hDetect, oriImgData.data, oriImgData.cols,
//                oriImgData.rows, face_box);
//            if (0 != retValue)
//                throw retValue;
//            /*retValue = save_max_rect_face(imgList[l], fd_mtcnn, face_box);
//            if (0 != retValue)
//                continue;*/
//
//            /*float feaPoints[10];
//            for (int j = 0; j < 5; ++j) {
//                feaPoints[2 * j] = face_box.ppoint[j];
//                feaPoints[2 * j + 1] = face_box.ppoint[j + 5];
//            }
//
//            cv::Mat oriImgData = cv::imread(imgList[l], CV_LOAD_IMAGE_COLOR);
//
//            cv::Mat cvt_image;
//            cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2RGB);
//
//            int age = 0;
//            float gender_score = 0.0f;
//            retValue = GetFaceBeautyScore(hFace, feaPoints,
//                cvt_image.data, oriImgData.cols, oriImgData.rows,
//                oriImgData.channels(), gender_score, age);*/
//            int age = 0;
//            float gender_score = 0.0f;
//            float beauty_score = 0.0f;
//            float glass_score = 0.0f;
//            float happy_score = 0.0f;
//            int emotion = 0;
//            //for (int test_idx = 0; test_idx < 1; ++test_idx) {
//            timeCount.Start();
//            retValue = DNHPXGetFaceGenderScore(hFace, face_box.key_points,
//                oriImgData.data, oriImgData.cols, oriImgData.rows,
//                oriImgData.channels(), gender_score, age, beauty_score,
//                glass_score, emotion, happy_score);
//            timeCount.Stop();
//            std::cout << "Attribute: " << 1000 * timeCount.GetTime() << "ms" << std::endl;
//
//            std::cout << imgList[l] << std::endl;
//            // ???????
//            if (gender_score > 0.5f)
//                std::cout << "Gender: female" << std::endl;
//            else
//                std::cout << "Gender: male" << std::endl;
//
//            // ????????
//            if (age <= 16)
//                std::cout << "child" << std::endl;
//            else if (age >= 62)
//                std::cout << "old" << std::endl;
//            else
//                std::cout << "Age: " << age << std::endl;
//
//            std::cout << "Beauty: " << beauty_score << std::endl;
//
//            if (glass_score > 0.5f)
//                std::cout << "No glasses" << std::endl;
//            else
//                std::cout << "Wear glasses" << std::endl;
//
//            if (0 == emotion)
//                std::cout << "angery" << std::endl;
//            else if (1 == emotion)
//                std::cout << "disgusted" << std::endl;
//            else if (2 == emotion)
//                std::cout << "fearful" << std::endl;
//            else if (3 == emotion)
//                std::cout << "happy, score: " << happy_score << std::endl;
//            else if (4 == emotion)
//                std::cout << "neutral" << std::endl;
//            else if (5 == emotion)
//                std::cout << "sad" << std::endl;
//            else if (6 == emotion)
//                std::cout << "surprise" << std::endl;
//        }
//#endif
        
        DNHPXUninitFaceGender(hFace);
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


