// Copyright 2020-present pytorch-cpp Authors
#include "vae.h"
#include <utility>
#include "api.h"

VAEImpl::VAEImpl(int64_t h_dim, int64_t output1, int64_t output2, int64_t output3, int64_t output4, int64_t z_dim)
    : fc1(output4, z_dim),
      fc2(output4, z_dim),
      fc3(z_dim, output4),
      cnv1(torch::nn::Conv2dOptions(1, output1, 4).stride(2)),//28->13
      cnv2(torch::nn::Conv2dOptions(output1, output2, 4).stride(2)),//5
      cnv3(torch::nn::Conv2dOptions(output2, output3, 4).stride(2).padding(1)),//2
      cnv4(torch::nn::Conv2dOptions(output3, output4, 4).stride(2).padding(1)),
      flt(torch::nn::FlattenOptions().start_dim(1)),
      //unflt(torch::nn::UnflattenOptions(1, {output4, h_dim, h_dim}))
      uncnv4(torch::nn::ConvTranspose2dOptions(output4, output3, 2).stride(2)),//2
      uncnv3(torch::nn::ConvTranspose2dOptions(output3, output2, 3).stride(2)),//5
      uncnv2(torch::nn::ConvTranspose2dOptions(output2, output1, 4).stride(2)),//12
      uncnv1(torch::nn::ConvTranspose2dOptions(output1, 1, 6).stride(2)),//28
      output4_(output4)
      {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("cnv1", cnv1);
    register_module("cnv2", cnv2);
    register_module("cnv3", cnv3);
    register_module("cnv4", cnv4);
    register_module("uncnv1", uncnv1);
    register_module("uncnv2", uncnv2);
    register_module("uncnv3", uncnv3);
    register_module("uncnv4", uncnv4);
    register_module("flt", flt);
    //register_module("unflt", unflt);
}

std::pair<torch::Tensor, torch::Tensor> VAEImpl::encode(torch::Tensor x) {
    x = torch::relu(cnv1(x));
    x = torch::relu(cnv2(x));
    x = torch::relu(cnv3(x));
    x = torch::relu(cnv4(x));
    x = flt(x);
    return {fc1->forward(x), fc2->forward(x)};
}

torch::Tensor VAEImpl::reparameterize(torch::Tensor mu, torch::Tensor log_var) {
    if (is_training()) {
        auto std = log_var.div(2).exp_();
        auto eps = torch::randn_like(std);
        return eps.mul(std).add_(mu);
    } else {
        // During inference, return mean of the learned distribution
        // See https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
        return mu;
    }
}

torch::Tensor VAEImpl::decode(torch::Tensor z) {
    auto h = torch::nn::functional::relu(fc3->forward(z));
    h = h.view({-1,output4_,1,1});
    h = torch::relu(uncnv4(h));
    h = torch::relu(uncnv3(h));
    h = torch::relu(uncnv2(h));
    h = torch::sigmoid(uncnv1(h));
    return h;
}

VAEOutput VAEImpl::forward(torch::Tensor x) {
    auto encode_output = encode(x);
    auto mu = encode_output.first;
    auto log_var = encode_output.second;
    auto z = reparameterize(mu, log_var);
    auto x_reconstructed = decode(z);
    return {x_reconstructed, mu, log_var};
}

static std::unique_ptr<VAE> vae_model;
static std::unique_ptr<torch::optim::Adam> vae_optimizer;
static int vae_index = 0;
static int vae_image_size;

void initialise_model_optimizer(int image_size, int h_dim, int z_dim, float learning_rate){
    vae_image_size = image_size;
    vae_model = std::make_unique<VAE>(1, 32, 64, 128, 256, z_dim);
    torch::Device device(torch::kCPU);
    (*vae_model)->to(device);
    vae_optimizer = std::make_unique<torch::optim::Adam>((*vae_model)->parameters(), torch::optim::AdamOptions(learning_rate));
}

int train_a_batch(std::vector<float> &data){
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int batch_num = data.size()/(vae_image_size*vae_image_size);
    std::vector<long int> dimensions = {batch_num, 1, vae_image_size, vae_image_size};
    torch::Tensor torch_real_projections = torch::from_blob(data.data(), c10::ArrayRef<long int>(dimensions), options);
    auto output = (*(vae_model))->forward(torch_real_projections);
    //compute loss or consider apply ctf
    auto reconstruction_loss = torch::nn::functional::mse_loss(output.reconstruction, torch_real_projections, torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum));
    auto kl_divergence = -0.5 * torch::sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp());
    auto loss = reconstruction_loss + kl_divergence;
    vae_optimizer->zero_grad();
    loss.backward();
    vae_optimizer->step();
    //save some reconstructions here
    if(vae_index % 100){
        //std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
        std::cout << vae_index + 1 << "/"
            << "Reconstruction loss: "
            << reconstruction_loss.item<double>() / torch_real_projections.size(0)
            << ", KL-divergence: " << kl_divergence.item<double>() / torch_real_projections.size(0)
            << std::endl;
    }

}
