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
        torch::nn::Sequential cnv1, cnv2, cnv3, cnv4, cnv5;
        torch::nn::Sequential uncnv1, uncnv2, uncnv3, uncnv4, uncnv5;
        torch::nn::BatchNorm2d bn;//, bn2, bn3, bn4, unbn1, unbn2, unbn3, unbn4;
        torch::nn::Flatten flt;
        int64_t output4_;
        //torch::nn::Unflatten unflt;
        torch::nn::Sequential cnvBlock(int inChannels, int outChannels, int kernelSize, int strideSize, int paddingSize = 0){
            return torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).stride(strideSize).padding(paddingSize).bias(false)),
                    torch::nn::BatchNorm2d(outChannels),
                    torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
                    );
        }
        torch::nn::Sequential uncnvBlock(int inChannels, int outChannels, int kernelSize, int strideSize, int paddingSize = 0){
            return torch::nn::Sequential(
                    torch::nn::BatchNorm2d(inChannels),
                    torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(inChannels, outChannels, kernelSize).stride(strideSize).padding(paddingSize).bias(false))
                    );
        }
};

TORCH_MODULE(VAE);

struct CUNet2dImpl : torch::nn::Module {
    CUNet2dImpl(int32_t inChannels, int32_t outChannels, int32_t inputSize, int32_t featureChannels=64, int32_t levels=5, int32_t kernelSize=3, bool wrapping=true, bool padding=true, bool batchNorm=false, bool convolutionDownsampling=true, bool convolutionUpsampling=true, bool partialConvolution=false, bool showSizes=false)
    {
        this->levels=levels;
        this->kernelSize=kernelSize;
        this->paddingSize=padding?(kernelSize-1)/2:0;
        this->inputSize = inputSize;

        this->convolutionDownsampling=convolutionDownsampling;
        this->convolutionUpsampling=convolutionUpsampling;
        this->partialConvolution=partialConvolution;
        this->batchNorm=batchNorm;
        this->showSizes=showSizes;
        this->wrapping=wrapping;

        for(int level=0; level<levels-1; level++)
        {
            contracting.push_back(levelBlock(level==0?inChannels:featureChannels*(1<<(level-1)), featureChannels*(1<<level)));
            register_module("contractingBlock"+std::to_string(level),contracting.back());

            downsampling.push_back(downsamplingBlock(featureChannels*(1<<level)));
            register_module("downsampling"+std::to_string(level),downsampling.back());
        }

        bottleneck=levelBlock(featureChannels*(1<<(levels-2)), featureChannels*(1<<(levels-1)));
        register_module("bottleneck",bottleneck);

        for(int level=levels-2; level>=0; level--)
        {
            upsampling.push_front(upsamplingBlock(featureChannels*(1<<(level+1)), featureChannels*(1<<level)));
            register_module("upsampling"+std::to_string(level),upsampling.front());

            expanding.push_front(levelBlock(featureChannels*(1<<level)*2, featureChannels*(1<<level)));
            register_module("expandingBlock"+std::to_string(level),expanding.front());
        }

        output=torch::nn::Conv2d(torch::nn::Conv2dOptions(featureChannels, outChannels, 1));
        register_module("output",output);

        if(this->wrapping)
        {
            initializeGrid(inputSize);
            flow = flowBlock(outChannels);
            register_module("flow", flow);
        }
    }

    void initializeGrid(int inputSize_) {
        torch::Tensor x = torch::arange(0, inputSize_, torch::dtype(torch::kFloat32));
        auto grids = torch::meshgrid({x, x});
        grid = torch::stack(grids);
        grid = grid.unsqueeze(0);
    }

    torch::Tensor forward(const torch::Tensor& inputTensor) {
        std::vector<torch::Tensor> contractingTensor(levels-1);
        std::vector<torch::Tensor> downsamplingTensor(levels-1);
        torch::Tensor bottleneckTensor;
        std::vector<torch::Tensor> upsamplingTensor(levels-1);
        std::vector<torch::Tensor> expandingTensor(levels-1);
        torch::Tensor outputTensor;
        torch::Tensor flowTensor;

        for(int level=0; level<levels-1; level++)
        {
            contractingTensor[level]=contracting[level]->forward(level==0?inputTensor:downsamplingTensor[level-1]);
            downsamplingTensor[level]=downsampling[level]->forward(contractingTensor[level]);
        }

        bottleneckTensor=bottleneck->forward(downsamplingTensor.back());

        for(int level=levels-2; level>=0; level--)
        {
            upsamplingTensor[level]=upsampling[level]->forward(level==levels-2?bottleneckTensor:expandingTensor[level+1]);
            if(paddingSize==0) { //apply cropping to the contracting tensor in order to concatenate with the same-level expanding tensor
                int oldXSize=contractingTensor[level].size(2);
                int oldYSize=contractingTensor[level].size(3);
                int newXSize=upsamplingTensor[level].size(2);
                int newYSize=upsamplingTensor[level].size(3);
                int startX=oldXSize/2-newXSize/2;
                int startY=oldYSize/2-newYSize/2;
                contractingTensor[level]=contractingTensor[level].slice(2,startX,startX+newXSize);
                contractingTensor[level]=contractingTensor[level].slice(3,startY,startY+newYSize);
            }
            expandingTensor[level]=expanding[level]->forward(torch::cat({contractingTensor[level],upsamplingTensor[level]},1));
        }

        outputTensor=output->forward(expandingTensor.front());

        if(showSizes)
        {
            std::cout << "input:  " << inputTensor.sizes() << std::endl;
            for(int level=0; level<levels-1; level++)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " contracting" << level << ":  " << contractingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " downsampling" << level << ": " << downsamplingTensor[level].sizes() << std::endl;
            }
            for(int i=0; i<levels-1; i++) std::cout << " "; std::cout << " bottleneck:    " << bottleneckTensor.sizes() << std::endl;
            for(int level=levels-2; level>=0; level--)
            {
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " upsampling" << level << ":   " << upsamplingTensor[level].sizes() << std::endl;
                for(int i=0; i<level; i++) std::cout << " "; std::cout << " expanding" << level << ":    " << expandingTensor[level].sizes() << std::endl;
            }
            std::cout << "output: " << outputTensor.sizes() << std::endl;
            showSizes=false;
        }

        if(wrapping)
        {
            flowTensor = flow->forward(outputTensor);
            flowTensor += grid;
            //assume the input is square
            flowTensor = 2.*(flowTensor/(float(inputSize) - 1.) - 0.5);
            //move channel dims to the last dim
            flowTensor = flowTensor.permute({0,2,3,1});
            //reverse last channel, not sure why, follow voxelmorph
            //flowTensor = at::flip(flowTensor, 3);
            return flowTensor;
        }

        return outputTensor;
    }

    //the 2d tensor size you pass to the model must be a multiple of this
    int sizeMultiple() {return 1<<(levels-1);}
    private:
    torch::nn::Sequential levelBlock(int inChannels, int outChannels)
    {
        if(batchNorm)
        {
            //if(partialConvolution)
            //    return torch::nn::Sequential(
            //            PartialConv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
            //            torch::nn::BatchNorm2d(outChannels),
            //            torch::nn::ReLU(),
            //            PartialConv2d(torch::nn::Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
            //            torch::nn::BatchNorm2d(outChannels),
            //            torch::nn::ReLU()
            //            );
            //else
            return torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
                    torch::nn::BatchNorm2d(outChannels),
                    torch::nn::ReLU(),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels, outChannels,
                            kernelSize).padding(paddingSize)),
                    torch::nn::BatchNorm2d(outChannels),
                    torch::nn::ReLU()
                    );
        } else
        {
            //if(partialConvolution)
            //    return
            //        torch::nn::Sequential(
            //                PartialConv2d(torch::nn::Conv2dOptions(inChannels,
            //                        outChannels,
            //                        kernelSize).padding(paddingSize)),
            //                torch::nn::ReLU(),
            //                PartialConv2d(torch::nn::Conv2dOptions(outChannels,
            //                        outChannels,
            //                        kernelSize).padding(paddingSize)),
            //                torch::nn::ReLU()
            //                );
            //else
            return torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels,
                            outChannels,
                            kernelSize).padding(paddingSize)),
                    torch::nn::ReLU(),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(outChannels,
                            outChannels,
                            kernelSize).padding(paddingSize)),
                    torch::nn::ReLU()
                    );
        }
    }

    torch::nn::Sequential downsamplingBlock(int channels)
    {
        if(convolutionDownsampling)
        {
            //if(partialConvolution)
            //    return
            //        torch::nn::Sequential(
            //                PartialConv2d(torch::nn::Conv2dOptions(channels,
            //                        channels,
            //                        kernelSize).stride(2).padding(paddingSize))
            //                );
            //else
            return torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(channels,
                            channels,
                            kernelSize).stride(2).padding(paddingSize))
                    );
        }
        else
        {
            return torch::nn::Sequential(
                    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
                    );
        }
    }

    torch::nn::Sequential upsamplingBlock(int inChannels, int outChannels)
    {
        if(convolutionUpsampling)
        {
            //if(partialConvolution)
            //    return
            //        torch::nn::Sequential(
            //                torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,
            //                            2})).mode(torch::kNearest)),
            //                PartialConv2d(torch::nn::Conv2dOptions(inChannels,
            //                        outChannels,
            //                        kernelSize).padding(paddingSize))
            //                );
            //else
            return torch::nn::Sequential(
                    torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,
                                2})).mode(torch::kNearest)),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels,
                            outChannels,
                            kernelSize).padding(paddingSize))
                    );
        }
        else
        {
            return torch::nn::Sequential(
                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(inChannels,
                            outChannels,
                            2).stride(2))
                    );
        }
    }

    torch::nn::Sequential flowBlock(int inChannels)
    {
        return torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels,
                        2,
                        1))
                );
    }

    int    levels;
    int    kernelSize;
    int    paddingSize;
    int    inputSize;

    bool   convolutionDownsampling;
    bool   convolutionUpsampling;
    bool   partialConvolution;
    bool   batchNorm;
    bool   showSizes;
    bool   wrapping;

    std::deque<torch::nn::Sequential>   contracting;
    std::deque<torch::nn::Sequential>   downsampling;
    torch::nn::Sequential               bottleneck;
    std::deque<torch::nn::Sequential>   upsampling;
    std::deque<torch::nn::Sequential>   expanding;
    torch::nn::Conv2d                   output{nullptr};
    torch::nn::Sequential               flow;
    torch::Tensor                       grid;
};
TORCH_MODULE_IMPL(CUNet2d, CUNet2dImpl);


