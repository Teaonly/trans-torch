#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#include <TH/TH.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

extern "C"
{
void* loadCaffeNet(const char* param_file, const char* model_file);
void releaseCaffeNet(void* net_);
void saveCaffeNet(void* net_, const char* weight_file);

void writeCaffeConvLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias);
void writeCaffeLinearLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias);
void writeCaffeBNLayer(void* net, const char* layername, THFloatTensor* mean, THFloatTensor* var);
void writeCaffeScaleLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias);
}

typedef float Dtype;

using namespace caffe;  // NOLINT(build/namespaces)

void* loadCaffeNet(const char* param_file, const char* model_file) {
  Net<Dtype>* net = new Net<Dtype>(string(param_file), TEST);
  if(model_file != NULL)
    net->CopyTrainedLayersFrom(string(model_file));

  return net;
}

void releaseCaffeNet(void* net_) {
    Net<Dtype>* net = (Net<Dtype>*)net_;
    if ( net != NULL) {
        delete net;
    }
}

void saveCaffeNet(void* net_, const char* weight_file) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    NetParameter net_param;
    net->ToProto(&net_param);

    WriteProtoToBinaryFile(net_param, std::string(weight_file));
}

int getTHTensorSize(THFloatTensor* tensor) {
    int size = tensor->size[0];
    for (int i = 1; i < tensor->nDimension; i++) {
        size = size *  tensor->size[i];
    }
    return size;
}

void writeCaffeBNLayer(void* net_, const char* layerName, THFloatTensor* mean, THFloatTensor* var) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    const boost::shared_ptr<caffe::Layer<Dtype> > inLayer = net->layer_by_name(std::string(layerName));
    vector<shared_ptr<Blob<Dtype> > > blobs = inLayer->blobs();

    // Checking size
    CHECK_EQ(blobs.size(), 3);
    CHECK_EQ(getTHTensorSize(mean), blobs[0]->count());

    // Converting 2 parameter(Torch) to 3 parameter(Caffe)
    const float* mean_ptr = THFloatTensor_data(mean);
    const float* var_ptr = THFloatTensor_data(var);

    caffe_set(blobs[2]->count(), 1.0f, blobs[2]->mutable_cpu_data());
    caffe_copy(blobs[0]->count(), mean_ptr, blobs[0]->mutable_cpu_data());
    caffe_copy(blobs[1]->count(), var_ptr, blobs[1]->mutable_cpu_data());
}


void writeCaffeScaleLayer(void* net_, const char* layerName, THFloatTensor* weights, THFloatTensor* bias) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    const boost::shared_ptr<caffe::Layer<Dtype> > inLayer = net->layer_by_name(std::string(layerName));
    vector<shared_ptr<Blob<Dtype> > > blobs = inLayer->blobs();

    // Checking size
    CHECK_EQ(blobs.size(), 2);
    CHECK_EQ(getTHTensorSize(weights), blobs[0]->count());

    // Copying data
    const float* data_ptr = THFloatTensor_data(weights);
    caffe_copy(blobs[0]->count(), data_ptr, blobs[0]->mutable_cpu_data());

    data_ptr = THFloatTensor_data(bias);
    caffe_copy(blobs[1]->count(), data_ptr, blobs[1]->mutable_cpu_data());
}

void writeCaffeConvLayer(void* net_, const char* layerName, THFloatTensor* weights, THFloatTensor* bias) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    const boost::shared_ptr<caffe::Layer<Dtype> > inLayer = net->layer_by_name(std::string(layerName));
    vector<shared_ptr<Blob<Dtype> > > blobs = inLayer->blobs();

    // Checking output layer is conv, so parameter's blob size is 2
    if ( blobs.size() != 2) {
        std::ostringstream oss;
        oss << "Can't write into layer :" << layerName ;
        THError(oss.str().c_str());
    }

    // Checking size
    CHECK_EQ(getTHTensorSize(weights), blobs[0]->count());
    CHECK_EQ(getTHTensorSize(bias), blobs[1]->count());

    // Copying data
    const float* data_ptr = THFloatTensor_data(weights);
    caffe_copy(blobs[0]->count(), data_ptr, blobs[0]->mutable_cpu_data());

    data_ptr = THFloatTensor_data(bias);
    caffe_copy(blobs[1]->count(), data_ptr, blobs[1]->mutable_cpu_data());
}

void writeCaffeLinearLayer(void* net_, const char* layerName, THFloatTensor* weights, THFloatTensor* bias) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    const boost::shared_ptr<caffe::Layer<Dtype> > inLayer = net->layer_by_name(std::string(layerName));
    vector<shared_ptr<Blob<Dtype> > > blobs = inLayer->blobs();

    // Checking output layer is conv, so parameter's blob size is 2
    if ( blobs.size() != 2) {
        std::ostringstream oss;
        oss << "Can't write into layer :" << layerName ;
        THError(oss.str().c_str());
    }

    // Checking size
    unsigned int th_weights_size = weights->size[0] * weights->size[1];
    CHECK_EQ(th_weights_size, blobs[0]->count());

    unsigned int th_bias_size = bias->size[0];
    CHECK_EQ(th_bias_size, blobs[1]->count());

    // Copying data
    const float* data_ptr = THFloatTensor_data(weights);
    caffe_copy(blobs[0]->count(), data_ptr, blobs[0]->mutable_cpu_data());

    data_ptr = THFloatTensor_data(bias);
    caffe_copy(blobs[1]->count(), data_ptr, blobs[1]->mutable_cpu_data());
}

