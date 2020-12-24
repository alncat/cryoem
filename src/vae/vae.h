// Copyright 2020-present pytorch-cpp Authors
#pragma once
#undef CUDA
#include <torch/torch.h>
#include <utility>

struct VAEOutput {
    torch::Tensor reconstruction;
    torch::Tensor mu;
    torch::Tensor log_var;
};

class VAEImpl : public torch::nn::Module {
 public:
    VAEImpl(int64_t h_dim, int64_t output1, int64_t output2, int64_t output3, int64_t output4, int64_t z_dim);
    torch::Tensor decode(torch::Tensor z);
    VAEOutput forward(torch::Tensor x);
 private:
    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x);
    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var);

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Conv2d cnv1, cnv2, cnv3, cnv4, cnv5;
    torch::nn::ConvTranspose2d uncnv1, uncnv2, uncnv3, uncnv4, uncnv5;
    //torch::nn::BatchNorm2d bn1, bn2, bn3, bn4, unbn1, unbn2, unbn3, unbn4;
    torch::nn::Flatten flt;
    int64_t output4_;
    //torch::nn::Unflatten unflt;
};

TORCH_MODULE(VAE);

