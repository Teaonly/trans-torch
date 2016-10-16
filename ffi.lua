local ffi = require 'ffi'

ffi.cdef[[
void* loadCaffeNet(const char* param_file, const char* model_file);
void releaseCaffeNet(void* net);
void saveCaffeNet(void* net_, const char* weight_file);

void writeCaffeLinearLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias); 
void writeCaffeConvLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias); 
void writeCaffeBNLayer(void* net, const char* layername, THFloatTensor* mean, THFloatTensor* var);
void writeCaffeScaleLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias);
]]

transTorch._C = ffi.load(package.searchpath('libtrans_torch', package.cpath))


