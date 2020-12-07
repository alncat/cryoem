

# OPUS-SSRI


OPUS-SSRI(Sparsity and Smoothness Regularized Imaging) is a stand-alone computer 
program for Maximum A Posteriori refinement of (multiple) 3D reconstructions in cryo-electron microscopy. It is developed in the 
research group of Jianpeng Ma in Baylor College of Medicine. This implementation is based on the software [RELION](https://www.ncbi.nlm.nih.gov/pubmed/22100448).

## Installation

OPUS-SSRI can be installed by using the docker command below.

```
docker pull alncat/opus-ssri:first
```
Another important step is setting up gpu support for docker. 
You can follow the instruction at https://www.tensorflow.org/install/docker .
We can then run the program in the docker image via
```
sudo docker run --name ssri-test --gpus all  -it -v /data:/data alncat/opus-ssri:first bash
```
You can later keep the source in docker image up to date by replacing the /relion-luo/src directory with the src in this repository and recompling the whole program using make in /relion-luo/build. The updated program now can be found in /relion-luo/build/bin.

## Usage

OPUS-SSRI introduces several more options to the 3D refinement program of RELION.
OPUS-SSRI solves the 3D reconstruction problem in the maximization step with penalty function of the form,

![\alpha\frac{|x_j|}{(|x_j^i|+\epsilon)} + \beta\frac{\|\nabla x_j\|_2}{(\|\nabla x_j^i\|_2\+\epsilon^')} + \gamma\|x_j-x_j^i \|^2](https://render.githubusercontent.com/render/math?math=%5Calpha%5Cfrac%7B%7Cx_j%7C%7D%7B(%7Cx_j%5Ei%7C%2B%5Cepsilon)%7D%20%2B%20%5Cbeta%5Cfrac%7B%5C%7C%5Cnabla%20x_j%5C%7C_2%7D%7B(%5C%7C%5Cnabla%20x_j%5Ei%5C%7C_2%5C%2B%5Cepsilon%5E')%7D%20%2B%20%5Cgamma%5C%7Cx_j-x_j%5Ei%20%5C%7C%5E2)

or in text format,

```
α|x_j|/(|x_j^i|+ϵ) + β‖∇x_j‖_2/(‖∇x_j^i‖_2+ϵ') + γ‖x_j-x_j^i ‖^2.
```

The new options are:

Option | Function
------------ | -------------
--tv |toggle on the OPUS-SSRI 3D refinement protocol
--tv_eps |the ![\epsilon](https://render.githubusercontent.com/render/math?math=%5Cepsilon) in the above equation
--tv_epsp |the ![\epsilon^'](https://render.githubusercontent.com/render/math?math=%5Cepsilon%5E') in the above equation
--tv_alpha |propotional to the ![\alpha\epsilon](https://render.githubusercontent.com/render/math?math=%5Calpha) in the above equation
--tv_beta |propotional to the ![\beta\epsilon^'](https://render.githubusercontent.com/render/math?math=%5Cbeta) in the above equation
--tv_weight |propotional to the ![\gamma](https://render.githubusercontent.com/render/math?math=%5Cgamma) in the above equation
--tv_iters |the number of iterations of the optimization algorithm
--tv_lr |propotional to the learning rate of the optimization algorithm

An exmaple of the command for running OPUS-SSRI is shown below,

```
mpiexec -n 3 /relion-luo/build/bin/relion_refine_mpi --o /output-folder --i particle-stack --ini_high 40 \ 
--dont_combine_weights_via_disc --pool 4 --ctf --ctf_corrected_ref --iter 25 --particle_diameter 256 \
--flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 \ 
--norm --scale --j 8 --gpu 0,1,2,3 --tv_alpha 1.0 --tv_beta 2.0 --tv_eps 0.01 --tv_epsp 0.01 \
--tv_weight 0.1 --tv_lr 0.5 --tv_iters 150 --ref initial-map --free_gpu_memory 256 --auto_refine \ 
--split_random_halves --low_resol_join_halves 50 --tv --adaptive_fraction 0.94 --preread_images --sym C4 

```
## Build instruction (Under development!!!)
To build this program from scratch on a ubuntu 16.04 machine, we can first create a build directory. You need to have a cmake with version above 3.15.2, a cuda 10.1, cudnn and latest version of libtorch. You also need to have an fftw library with threads. In case of missing reference to cublas, we can install cublas and link it to the directory of cuda 10.1 manually by executing 
```
sudo ln -s -T /usr/lib/x86_64-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so
```
We then change to the build directory. Inside the build directory, execute
```
/usr/local/bin/cmake -DCMAKE_INSTALL_PREFIX=/fullpathof/build/bin/ -DCMAKE_BUILD_TYPE=relwithdebinfo -DCMAKE_C_COMPILER=gcc-5 -DCMAKE_CXX_COMPILER=g++-5 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 -DCUDNN_INCLUDE_DIR=/usr/include -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu -DCMAKE_PREFIX_PATH=/fullpathof/libtorch/share/cmake/Torch ..
```
Remeber to substitute the fullpathof with your complete path.
After this, execute
```
make
```
then
```
make install
```
You can have program inside build/bin .

The reconstructed results of OPUS-SSRI for some systems can be accessed at https://www.dropbox.com/sh/8ln07s9esmnnvhe/AADRk4UddUyfTFTa0KFK1NvYa?dl=0 .
