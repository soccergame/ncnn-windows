// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include "platform.h"
#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#include "command.h"
#include "pipeline.h"
#endif // NCNN_VULKAN

namespace ncnn {

class Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

    // layer implementation specific setup
    // return 0 if success
    virtual int create_pipeline(const Option& opt = Option());

    // layer implementation specific clean
    // return 0 if success
    virtual int destroy_pipeline(const Option& opt = Option());

public:
    // one input and one output blob
    bool one_blob_only;

    // support inplace inference
    bool support_inplace;

    // support vulkan compute
    bool support_vulkan;

    // accept input blob with packed storage
    bool support_packing;

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt = Option()) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt = Option()) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt = Option()) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt = Option()) const;

#if NCNN_VULKAN
public:
    // upload weight blob from host to device
    virtual int upload_model(VkTransfer& cmd, const Option& opt = Option());

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt = Option()) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt = Option()) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt = Option()) const;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt = Option()) const;

public:
    // assigned immediately after creating this layer
    const VulkanDevice* vkdev;
#endif // NCNN_VULKAN

public:
    // layer type index
    int typeindex;
#if NCNN_STRING
    // layer type name
    std::string type;
    // layer name
    std::string name;
#endif // NCNN_STRING
    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;
};

// layer factory function
// 声明了一个函数指针，即 指向任意函数名，只要返回值类型和传入参数一致即可
// 这里typedef通过类型别名的方式，定义了一个函数指针：
// layer* (*layer_creator_func) ();
// 此后layer_creator_func这个指针可以指向任何返回值为layer*，传入参数为空的函数，而不必理会函数名
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
};

#if NCNN_STRING
// get layer type from type name
int layer_to_index(const char* type);
// create layer from type name
Layer* create_layer(const char* type);
#endif // NCNN_STRING
// create layer from layer type
Layer* create_layer(int index);

// # 将宏参数不经扩展地转换成字符串常量
// ## 标记连接操作
// 带参数的宏定义
#define DEFINE_LAYER_CREATOR(name) \
    ::ncnn::Layer* name##_layer_creator() { return new name; }

} // namespace ncnn

#endif // NCNN_LAYER_H
