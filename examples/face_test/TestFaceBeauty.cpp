#include "face_beauty.h"
#include "FaceDetectEngine.hpp"
#include "autoarray.h"
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <time.h>
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
    
    try
    {
        // Initialize        
        retValue = SetFaceBeautyLibPath(pModulePath);
        
        BeautyHandle hFace;
        retValue |= InitOLDFaceBeauty(0, 0, &hFace, 4);
        //retValue |= InitDeepFeat("NNModel.dat", gpuId, &hAge);
        if (0 != retValue)
            throw retValue;

        retValue = FaceDetect_setLibPath(pModulePath);
        SN::FDHANDLE hDetect;
        retValue |= FaceDetect_init(&hDetect);
        if (0 != retValue) {
            std::cout << "Error Code: " << retValue << std::endl;
            UninitFaceBeauty(hFace);
            throw retValue;
        }
        
        // Read Image
        //cv::Mat garyImgData = cv::imread(strImgName, CV_LOAD_IMAGE_GRAYSCALE);
#ifdef _WIN32
        cv::Mat oriImgData = cv::imread(strImgName, CV_LOAD_IMAGE_COLOR);
        // Face detection
        SN::DetectedFaceBox face_box;
        retValue = FaceDetect_maxDetect(hDetect, oriImgData, face_box);
        if (0 != retValue)
            throw retValue;
        
        cv::Mat cvt_image;
        cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2RGB);
        float beauty = 0.0f;
        retValue = GetFaceBeautyScore(hFace, face_box.keypoints,
            cvt_image.data, oriImgData.cols, oriImgData.rows,
            oriImgData.channels(), beauty);

        beauty = beauty * 1.11f;
        if (beauty > 100.0f)
            beauty = 100.0f;
            
        std::cout << "Total score: " << beauty << std::endl;
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

            cv::Mat cvt_image;
            cv::cvtColor(oriImgData, cvt_image, cv::COLOR_BGR2RGB);
            float beauty = 0.0f;
            retValue = GetFaceBeautyScore(hFace, face_box.keypoints,
                cvt_image.data, oriImgData.cols, oriImgData.rows,
                oriImgData.channels(), beauty);

            beauty = beauty * 1.11f;
            if (beauty > 100.0f)
                beauty = 100.0f;

            std::cout << imgList[l] << " score: " << beauty << std::endl;
        }
#endif
        
        //// 2、瑕疵
        //featDim = GetDeepFeatSize(hXiaci) / 4;
        //pFeatures.reset(featDim);
        //nRetCode = InnerDeepFeat(hXiaci, pCropNormFace, 1, 3, 256, 256, pFeatures);

        //float maxR = -10000.0f;
        //int label = 15;
        //for (int j = 0; j < featDim; ++j)
        //{
        //    //std::cout << pFeatures[j] << " ";
        //    if (maxR < pFeatures[j])
        //    {
        //        maxR = pFeatures[j];
        //        label = j;
        //    }
        //}

        //if (0 == label)
        //    std::cout << "The flaws' number: " << "none!" << std::endl;
        //else if (1 == label)
        //    std::cout << "The flaws' number: " << "a little!" << std::endl;
        //else if (2 == label)
        //    std::cout << "The flaws' number: " << "small!" << std::endl;
        //else if (3 == label)
        //    std::cout << "The flaws' number: " << "a lot!" << std::endl;
        //else if (4 == label)
        //    std::cout << "The flaws' number: " << "very much!" << std::endl;

        //// 3、开心
        //featDim = GetDeepFeatSize(hHappy) / 4;
        //pFeatures.reset(featDim);
        //nRetCode = InnerDeepFeat(hHappy, pCropNormFace, 1, 3, 256, 256, pFeatures);

        //maxR = -10000.0f;
        //label = 15;
        //for (int j = 0; j < featDim; ++j)
        //{
        //    //std::cout << pFeatures[j] << " ";
        //    if (maxR < pFeatures[j])
        //    {
        //        maxR = pFeatures[j];
        //        label = j;
        //    }
        //}
        //// std::cout << std::endl;

        //if (0 == label)
        //    std::cout << "Angry!" << std::endl;
        //else if (1 == label)
        //    std::cout << "Unhappy!" << std::endl;
        //else if (2 == label)
        //    std::cout << "normal!" << std::endl;
        //else if (3 == label)
        //    std::cout << "happy!" << std::endl;
        //else if (4 == label)
        //    std::cout << "smile!" << std::endl;

        // 4、年龄
        //featDim = GetDeepFeatSize(hAge) / 4;
        //pFeatures.resize(featDim);
        //retValue = InnerDeepFeat(hAge, pNormImage5Pt, 1, 3, 256, 256, pFeatures);
    
        //maxR = -10000.0f;
        //label = 15;
        //for (int j = 0; j < featDim; ++j)
        //{
        //    //std::cout << pFeatures[j] << " ";
        //    if (maxR < pFeatures[j])
        //    {
        //        maxR = pFeatures[j];
        //        label = j;
        //    }
        //}
        //if (0 == label)
        //    std::cout << "小孩!" << std::endl;
        //else if (1 == label)
        //    std::cout << "少年!" << std::endl;
        //else if (2 == label)
        //    std::cout << "青年!" << std::endl;
        //else if (3 == label)
        //    std::cout << "中年!" << std::endl;
        //else if (4 == label)
        //    std::cout << "老年!" << std::endl;
        // std::cout << std::endl;

        //// 5、肤色
        //featDim = GetDeepFeatSize(hSkin) / 4;
        //pFeatures.reset(featDim);
        //nRetCode = InnerDeepFeat(hSkin, pCropNormFace, 1, 3, 256, 256, pFeatures);

        //score = pFeatures[0] * 1.11f;
        //if (score > 100.0f)
        //    score = 100.0f;    
        //std::cout << "Skin score: " << score << std::endl;
        
        // Uninitialized
        //FaceDetectUninit();
        //FaceAlignmentUninit();
        UninitFaceBeauty(hFace);
        //UninitDeepFeat(hSkin);
        //UninitDeepFeat(hXiaci);
        //UninitDeepFeat(hHappy);
        //UninitDeepFeat(hAge);
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


