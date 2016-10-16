// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_binary v0_net_proto_file_in net_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::ofstream;
using namespace caffe;  // NOLINT(build/namespaces)

typedef float Dtype;

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;  // Print output to stderr (while still logging)
  ::google::InitGoogleLogging(argv[0]);

  if ( argc < 3) {
    LOG(ERROR) << "Please input model file and weight file";
    return -1;
  }
  std::string model_file = argv[1];
  std::string weight_file = argv[2];

  Caffe::set_mode(Caffe::CPU);
  boost::shared_ptr<Net<Dtype> > net(new Net<Dtype>(model_file, caffe::TEST));
  net->CopyTrainedLayersFrom(weight_file);

  const boost::shared_ptr<Blob<Dtype> > inBlob = net->blob_by_name("data");
  Dtype* result = (Dtype*)(inBlob->cpu_data() + inBlob->offset(0));

  for (int i = 0; i < inBlob->count(); i++) {
    result[i] = 0.5;
  }

  net->Forward();

  const boost::shared_ptr<Blob<Dtype> > outBlob = net->blob_by_name("prob");
  result = (Dtype*)(outBlob->cpu_data() + outBlob->offset(0));

  for (int i = 0; i < outBlob->count(); i++) {
      std::cout << result[i] << " ";
  }
  std::cout << std::endl;
}
