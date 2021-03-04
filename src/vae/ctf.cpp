#include "ctf.h"
#include <vector>

void optimize_ctf(std::vector<float>& image_data, std::vector<float>& ref_data, float& delta_u, float& delta_v, float& angle, float defocus_res, std::vector<float> Ks, float Q0, int image_size, float angpix, bool do_damping) {
    torch::Tensor deltau = torch::tensor(delta_u, torch::requires_grad());
    torch::Tensor deltav = torch::tensor(delta_v, torch::requires_grad());
    torch::Tensor deltaa = torch::tensor(angle, torch::requires_grad());
    
    std::vector<long int> dimensions = {image_size, image_size};
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor image = torch::from_blob(image_data.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor ref_image = torch::from_blob(ref_data.data(), c10::ArrayRef<long int>(dimensions), options);

    float xs = image_size*angpix;
    float ys = image_size*angpix;
    
    auto correlation_fn = [&](){
        torch::Tensor defocus_average = -(deltau + deltav)*0.5;
        torch::Tensor defocus_deviation = -(deltau - deltav)*0.5;
        torch::Tensor x_idx = torch::arange(image_size/2+1)/float(image_size);//torch::arange(0, 6);
        torch::Tensor y_idx = torch::arange(-image_size/2, image_size)/float(image_size);
        auto grid = torch::meshgrid({y_idx, x_idx});
        //grid[0] corresponding to y, where grid[1] corresponding to x
        gird[0] /= ys;
        grid[1] /= xs;
        grid[0] = torch::roll(grid[0], {image_size/2}, {0});
        torch::Tensor angle = torch::atan2(grid[0], grid[1]);
        torch::Tensor deltaf = defocus_average + defocus_deviation*torch::cos(2.*(angle - deltaa));
        //std::cout << grid[0] << std::endl;
        //std::cout << grid[1] << std::endl;
        torch::Tensor u2 = grid[0]*grid[0] + grid[1]*grid[1];
        torch::Tensor u4 = u2*u2;
        auto argument = Ks[0]*deltaf*u2 + Ks[1]*u4 - Ks[4];
        torch::Tensor ctf = -(Ks[2]*torch::sin(argument) - Q0*torch::cos(argument));
        if(do_damping) {
            auto scale = torch::exp(Ks[3] * u2);
            ctf *= scale;
        }
        //auto image_fft = torch::fft::rfftn(image);
        auto ref_image_centered = torch::roll(ref_image, {image_size/2, image_size/2}, {0, 1});
        auto ref_image_fft = torch::fft::rfftn(ref_image_centered);
        auto ref_image_fft_ctf = ref_image_fft*ctf;
        auto ref_image_ctf = torch::fft::irfftn(ref_image_fft_ctf);
        ref_image_ctf = torch::roll(ref_image_ctf, {image_size/2, image_size/2}, {0, 1});
        image = image.reshape({1,image_size*image_size});
        ref_image_ctf = ref_image_ctf.reshape({1,image_size*image_size});
        auto restraint = defocus_deviation*defocus_deviation/(defocus_res*defocus_res.*image_size*image_size);
        return -torch::nn::functional::cosine_similarity(image, ref_image_ctf);
    };
    torch::optim::LBFGS lbfgs_optimizer({deltau, deltav, deltaa}, torch::optim::LBFGSOptions().line_search_fn("strong_wolfe"));
    auto cost = [&](){
        lbfgs_optimizer.zero_grad();
        auto correlation = correlation_fn();//torch::sum(image*ref_image_ctf);
        std::cout << "correlation: " << correlation << std::endl;
        //correlation.backward({}, c10::optional<bool>(true));
        correlation.backward();
        std::cout << deltau << std::endl;
        std::cout << deltav << std::endl;
        std::cout << deltaa << std::endl;
        return correlation;
    };
    for (int i = 0; i < 2; i++)
        lbfgs_optimizer.step(cost);
}
