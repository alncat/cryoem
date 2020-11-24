#pragma once
#include <iostream>
#pragma push_macro("PI")
#undef PI
#include "tiny_dnn/tiny_dnn.h"

void deconv_ae(tiny_dnn::network<tiny_dnn::sequential> &nn,
               std::vector<tiny_dnn::label_t> train_labels,
               std::vector<tiny_dnn::label_t> test_labels,
               std::vector<tiny_dnn::vec_t> train_images,
               std::vector<tiny_dnn::vec_t> test_images);
#pragma pop_macro("PI")
