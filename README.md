
Our latest version can do CTF refinement with 3D refinement simultaneously!!! checkout [CTF refinement](#ctf)

#Compilation
You can compile this program with fftw in external directory!
First, modify the ```CMakeLists.txt``` file, replace the cuda architecture compatible with your GPU! and set the ```MPI_LIBRARIES, MPI_INCLUDE_PATH``` to the mpi related paths on your system!
We recommend using cmake >= 3.14, gnu8 and cuda 10.2 !
```
mkdir build & cd build
```
```
FFTW_LIB=/work/jpma/luo/ssri/ssri_remote/external/fftw/lib FFTW_INCLUDE=/work/jpma/luo/ssri/ssri_remote/external/fftw/include/ CC=`which mpicc` CXX=`which mpic++` MPI_HOME=/opt/ohpc/pub/apps/openmpi/4.0.2_cuda10.1/ cmake -DCMAKE_INSTALL_PREFIX=/work/jpma/luo/ssri/ssri_remote/build/bin/ -DCMAKE_BUILD_TYPE=relwithdebinfo -DGUI=OFF ..
```
Remember to replace the environment variables with your settings!

# OPUS-SSRI


OPUS-SSRI(Sparsity and Smoothness Regularized Imaging) is a stand-alone computer
program for Maximum A Posteriori refinement of (multiple) 3D reconstructions in cryo-electron microscopy. It is developed in the
research group of Jianpeng Ma in Baylor College of Medicine. This implementation is based on the software [RELION](https://www.ncbi.nlm.nih.gov/pubmed/22100448).

## Installation

OPUS-SSRI can be obtained as a docker image by executing command below.

```
docker pull alncat/opus-ssri:first
```
Another important step is setting up gpu support for docker.
You can follow the instruction at https://www.tensorflow.org/install/docker .
We can then create a docker container from the docker image via
```
sudo docker run --name ssri-test --gpus all  -it -v /data:/data alncat/opus-ssri:first bash
```
After entering the docker containter, the OPUS-SSRI program is located at ```/relion-luo/build/bin```. You can perform the 3D refinement using the program in that directory.

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

The command for running OPUS-SSRI is shown below,

```
mpiexec --allow-run-as-root -n 3 /relion-luo/build/bin/relion_refine_mpi --o /output-folder --i particle-stack --ini_high 40 \
--dont_combine_weights_via_disc --pool 4 --ctf --ctf_corrected_ref --iter 25 --particle_diameter 256 \
--flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 \
--norm --scale --j 8 --gpu 0,1,2,3 --tv_alpha 1.0 --tv_beta 2.0 --tv_eps 0.01 --tv_epsp 0.01 \
--tv_weight 0.1 --tv_lr 0.5 --tv_iters 150 --ref initial-map --free_gpu_memory 256 --auto_refine \
--split_random_halves --low_resol_join_halves 50 --tv --adaptive_fraction 0.94 --preread_images --sym C4

```
You should replace ```output-folder, particle-stack and initial-map``` with your own path and files, respectively. You should also set the ```particle_diameter and sym``` according to the parameters of your system. For very large datasets that cannot fit in the memory, you should remove the ```preread_images``` option. ```tv_eps``` and ```tv_epsp``` are two adjustable parameters. You can set them around the level of density as ```the threshold for creating mask```. You can also experiment with different ```tv_alpha```, ```tv_beta``` and ```tv_weight```.

## CTF refinement <a name="ctf"></a>

We added the capacity to perform per-particle CTF refinement in a new version. You can use this program to perform consensus 3D refinement with per-particle CTF refinement. The program can be obtained as a docker image by

```
docker pull alncat/ssri-torch:ctf1
```

To run the CTF refinement, you need to create a docker container using the downloaded image by

```
sudo docker run --name ssri-ctf --gpus all  -it -v /data:/data alncat/ssri-torch:ctf1 bash
```
```--name``` is the name of container, ```-v``` argument is used to bind directory in host to the directory in container, the first option before colon refers to the directory in host, the second option refers to the directory in the container.

The docker container can be accessed later via

```
sudo docker start ssri-ctf && sudo docker exec -it ssri-ctf bash
```

The compiled program with CTF refinement capability is located at ```/root/gpu/ssri_remote/build/bin``` in your docker container. And the source code can be found at ```/root/gpu/ssri_remote/src``` .
Note that the program is compiled against Nvidia GPU with compute capacity ```sm_61```. To execute this program on Nvidia GPUs with other CUDA architectures, you should recompile this program by changing all ```sm_61``` and ```compute_61``` variables in ```/root/gpu/ssri_remote/CMakeLists.txt``` to the corresponding architecture of your GPUs. You can then recompile the program by first loading the compilation toolset,

```
scl enable devtoolset-7 bash
```

then executing
```
cd build && make
```
The options for controlling the CTF refinement are listed below,

Option | Function
------------ | -------------
--ctf_order   | Default is 2, start CTF refinement when the healpix sampling order is larger than this value
--refine_ctf_angle | Add this option to your command to refine the defoucs angle
--ctf_defocus_dev | Default is 1.0, the restraint strength of the deviation of defocus values from previous refinement
--ctf_defocus_iso | Default is 0.5, the restraint strength of the difference between two defocus depths

An exmaple of the command for running OPUS-SSRI with CTF refinement is shown below,

```
mpiexec --allow-run-as-root -n 3 /root/gpu/ssri/remote/build/bin/relion_refine_mpi --o /output-folder --i particle-stack --ini_high 40 \
--dont_combine_weights_via_disc --pool 4 --ctf --ctf_corrected_ref --iter 25 --particle_diameter 256 \
--flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 \
--norm --scale --j 8 --gpu 0,1,2,3 --tv_alpha 1.0 --tv_beta 2.0 --tv_eps 0.01 --tv_epsp 0.01 \
--tv_weight 0.1 --tv_lr 0.5 --tv_iters 150 --ref initial-map --free_gpu_memory 256 --auto_refine \
--split_random_halves --low_resol_join_halves 50 --tv --adaptive_fraction 0.94 --preread_images --sym C4 --sigma2_fudge 0.5 --ctf_defocus_iso 0.5

```
You should replace ```output-folder, particle-stack and initial-map``` with your own path and files, respectively. You should also set the ```particle_diameter and sym``` according to the parameters of your system. For very large datasets that cannot fit in the memory, you should remove the ```preread_images``` option.
