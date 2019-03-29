#include "face_recognition.h"
#include "FaceDetectEngine.hpp"
#include "autoarray.h"
#include "ListOperation.h"
#include "MyString.h"
#ifdef _WIN32
#include "TimeCount.h"
#endif
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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
    if (argc != 5) {
        std::cout << "usage:" << std::endl;
        std::cout << "mtcnn_test <module_path> <base_path> <filelist> <output>" << std::endl;
        return -1;
    }
    std::string modulePath = argv[1];
    const char* pModulePath = modulePath.c_str();
        
    std::string base_path = argv[2];
    int retValue = 0;
    float maxR = 0;
    int label = 0;
#ifdef _WIN32
    CTimeCount timeCount;
#endif
    
    try
    {
        // Initialize        
        retValue = SetFaceRecognitionLibPath(pModulePath);
        
        RecognitionHandle hFace;
        retValue |= InitOLDFaceRecognition(0, 0, &hFace, 4);
        //retValue |= InitDeepFeat("NNModel.dat", gpuId, &hAge);
        if (0 != retValue) {
            std::cout << "Error Code: " << retValue << std::endl;
            throw retValue;
        }

        retValue = FaceDetect_setLibPath(pModulePath);
        SN::FDHANDLE hDetect;
        retValue |= FaceDetect_init(&hDetect);
        if (0 != retValue) {
            std::cout << "Error Code: " << retValue << std::endl;
            UninitFaceRecognition(hFace);
            throw retValue;
        }
            
        
        // Read Image
        //cv::Mat garyImgData = cv::imread(strImgName, CV_LOAD_IMAGE_GRAYSCALE);
#ifdef _WIN32
        CMyString filename = argv[3];
        std::vector<CMyString> imgList;
        std::vector<pair<CMyString, int>> vecClassNameSampleNum;
        std::vector<int> vecSampleToClassIndex;
        ReadFileList(filename, imgList, vecClassNameSampleNum, vecSampleToClassIndex);
        /*vector<string> imgList;
        std::ifstream infile(argv[3]);
        std::string line;
        while (std::getline(infile, line)) {
            imgList.push_back(line);
        }
        infile.close();*/

        std::ofstream outfile(argv[4]);
        outfile << "{";
        for (int i = 0; i < imgList.size(); ++i) {
            std::string strImgName = base_path + imgList[i].c_str();
            cv::Mat oriImgData = cv::imread(strImgName, CV_LOAD_IMAGE_COLOR);
            SN::DetectedFaceBox face_box;
            retValue = FaceDetect_maxDetect(hDetect, oriImgData, face_box);
            if (0 != retValue)
                continue;

            if (i > 0)
                outfile << ",";

            cv::Mat cvt_image;
            cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2RGB);

            float *feature = 0;
            int fea_dim = 0;

            retValue = GetFaceRecognitionFeature(hFace, face_box.keypoints,
                cvt_image.data, oriImgData.cols, oriImgData.rows,
                oriImgData.channels(), &feature, fea_dim);

            outfile << "\"" << imgList[i] << "\"" << ":[";
            outfile << feature[0];
            for (int j = 1; j < fea_dim; ++j)
                outfile << "," << feature[j];
            outfile << "]";

            delete[] feature;
            feature = 0;
        }
        outfile << "}";
        outfile.close();
        
        // Face detection
        //timeCount.Start();
        

        //timeCount.Stop();
        //std::cout << "Detection: " << 1000 * timeCount.GetTime() << "ms" << std::endl;
        
        

        
        //for (int test_idx = 0; test_idx < 1; ++test_idx) {
            //timeCount.Start();
        
            //timeCount.Stop();
            //std::cout << "[" << test_idx << "]Gender: " 
            //    << 1000 * timeCount.GetTime() << "ms" << std::endl;
        //}


        
#else
        char path_list[512] = { 0 };
        strcpy(path_list, strImgName.c_str());

        vector<string> imgList;
        readFileList(path_list, imgList);

        for (int l = 0; l < imgList.size(); l++)
        {
            cv::Mat oriImgData = cv::imread(imgList[l], CV_LOAD_IMAGE_COLOR);
            // Face detection
            SN::DetectedFaceBox face_box;
            retValue = FaceDetect_maxDetect(hDetect, oriImgData, face_box);
            if (0 != retValue)
                throw retValue;
            /*retValue = save_max_rect_face(imgList[l], fd_mtcnn, face_box);
            if (0 != retValue)
                continue;*/

            /*float feaPoints[10];
            for (int j = 0; j < 5; ++j) {
                feaPoints[2 * j] = face_box.ppoint[j];
                feaPoints[2 * j + 1] = face_box.ppoint[j + 5];
            }

            cv::Mat oriImgData = cv::imread(imgList[l], CV_LOAD_IMAGE_COLOR);

            cv::Mat cvt_image;
            cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2RGB);

            int age = 0;
            float gender_score = 0.0f;
            retValue = GetFaceBeautyScore(hFace, feaPoints,
                cvt_image.data, oriImgData.cols, oriImgData.rows,
                oriImgData.channels(), gender_score, age);*/
            cv::Mat cvt_image;
            cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2RGB);
            int age = 0;
            float gender_score = 0.0f;
            float beauty_score = 0.0f;
            retValue = GetFaceGenderScore(hFace, face_box.keypoints,
                cvt_image.data, oriImgData.cols, oriImgData.rows,
                oriImgData.channels(), gender_score, age, beauty_score);

            std::cout << imgList[l] << std::endl;
            // 计算性别
            if (gender_score > 0.5f)
                std::cout << "Gender: female" << std::endl;
            else
                std::cout << "Gender: male" << std::endl;

            // 计算年龄
            if (age <= 16)
                std::cout << "child" << std::endl;
            else if (age >= 62)
                std::cout << "old" << std::endl;
            else
                std::cout << "Age: " << age << std::endl;

            std::cout << "Beauty: " << beauty_score << std::endl;
        }
#endif
        
        UninitFaceRecognition(hFace);
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


