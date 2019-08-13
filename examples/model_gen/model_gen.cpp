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
#include "MyString.h"
#include "MyFile.h"
#include "tclap/CmdLine.h"
#include "AlgorithmUtils.h"

#ifndef _WIN32
#include<unistd.h> 
#include <dirent.h>
#endif

int _tmain(int argc, TCHAR *argv[])
{
    int retValue = 0;
    std::locale::global(std::locale(""));

    try
    {
        TCLAP::CmdLine cmd(TCLAP_TEXT("Generate or extract caffe models.\n")  \
            TCLAP_TEXT("Copyright: He Zhixiang.")	\
            TCLAP_TEXT("Author: He Zhixiang")	\
            TCLAP_TEXT("Data: Nov. 5, 2018"), TCLAP_TEXT(' '), TCLAP_TEXT("2.0"));

        // Deployed prototxt file path
        TCLAP::MultiArg<tstring> modelPaths(TCLAP_TEXT("M"),
            TCLAP_TEXT("ModelPathsFile"),
            TCLAP_TEXT("File includes generate prototxt and caffemodel or dat needs to be extracted"),
            true, TCLAP_TEXT("Common use for generate"), cmd);

        TCLAP::ValueArg<tstring> resultDir(TCLAP_TEXT(""),
            TCLAP_TEXT("ResultDir"),
            TCLAP_TEXT("Directiry to store results"),
            false, TCLAP_TEXT(""),
            TCLAP_TEXT("result directory"), cmd);

        cmd.parse(argc, argv);

        vector<tstring> model_paths = modelPaths.getValue();
        CMyString result_dir = resultDir.getValue();

        CMyString model_path;
        std::vector<CMyString> net_names, weight_names;
        for (auto iter = model_paths.cbegin(); iter != model_paths.cend(); iter = iter + 2)
        {
            net_names.push_back(*iter);
            weight_names.push_back(*(iter + 1));
        }

        const int ModelNumber = net_names.size();
        std::vector<int> solverLen, modelLen;
        for (int i = 0; i < ModelNumber; ++i)
        {
            std::fstream fileSolver;
            fileSolver.open(
                net_names[i].c_str(), std::fstream::in | std::fstream::binary);
            if (false == fileSolver.is_open())
                return 1;

            fileSolver.seekg(0, std::fstream::end);
            solverLen.push_back(int(fileSolver.tellg()));
            fileSolver.seekg(0, std::fstream::beg);

            std::fstream fileModel;
            fileModel.open(
                weight_names[i].c_str(), std::fstream::in | std::fstream::binary);
            if (false == fileModel.is_open())
                return 1;

            fileModel.seekg(0, std::fstream::end);
            modelLen.push_back(int(fileModel.tellg()));
            fileModel.seekg(0, std::fstream::beg);

            fileSolver.close();
            fileModel.close();
        }

        // 开始写入成为一个文件
        int totalLen = 0;
        for (int i = 0; i < ModelNumber; ++i)
            totalLen += solverLen[i] + modelLen[i] + sizeof(int) * 2;
        totalLen += sizeof(int);

        totalLen = ((totalLen + 7) / 8) * 8;
        AutoArray<char> tempBuffer(totalLen);
        int *pBuffer = reinterpret_cast<int *>(tempBuffer.begin());
        pBuffer[0] = ModelNumber;
        for (int i = 0; i < ModelNumber; ++i)
        {
            pBuffer[2 * i + 1] = solverLen[i];
            pBuffer[2 * i + 2] = modelLen[i];
        }
        char *pDataPtr = tempBuffer + sizeof(int) * (2 * ModelNumber + 1);

        for (int i = 0; i < ModelNumber; ++i)
        {
            std::fstream fileSolver;
            fileSolver.open(
                net_names[i].c_str(), 
                std::fstream::in | std::fstream::binary);
            if (false == fileSolver.is_open())
                return 1;

            fileSolver.read(pDataPtr, solverLen[i]);

            std::fstream fileModel;
            fileModel.open(
                weight_names[i].c_str(), 
                std::fstream::in | std::fstream::binary);
            if (false == fileModel.is_open())
                return 1;

            fileModel.read(pDataPtr + solverLen[i], modelLen[i]);

            fileSolver.close();
            fileModel.close();

            pDataPtr = pDataPtr + solverLen[i] + modelLen[i];
        }

        // 加密	
        int numOfData = totalLen / sizeof(pBuffer[0]);
        for (int i = 0; i < numOfData; ++i)
        {
            int tempData = pBuffer[i];
            pBuffer[i] = hzx::rol(
                static_cast<unsigned int>(tempData), hzx::g_shiftBits);
        }

        // write encrypted model file	
#ifdef _UNICODE
        model_path.Format(_T("%ls/merge_bin.dat"), result_dir.c_str());
#else
        model_path.Format(_T("%s/merge_bin.dat"), result_dir.c_str());
#endif
        /*CMyFile fileResult(model_path, CMyFile::modeCreate | CMyFile::modeWrite);
        fileResult.Write(tempBuffer, totalLen);
        fileResult.Close();*/
        std::fstream fileResult;
        fileResult.open(
            model_path.c_str(), std::fstream::out | std::fstream::binary);
        if (false == fileResult.is_open())
            return 1;
        fileResult.write(tempBuffer.begin(), totalLen);
        fileResult.close();   
    }
    catch (int errcode)
    {
        retValue = errcode;
    }
    catch (const std::bad_alloc &)
    {
        retValue = -1;
    }
    catch (...)
    {
        retValue = -2;
    }

    return retValue;
}


