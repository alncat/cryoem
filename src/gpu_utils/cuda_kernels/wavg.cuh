#ifndef CUDA_WAVG_KERNEL_CUH_
#define CUDA_WAVG_KERNEL_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "src/gpu_utils/cuda_projector.cuh"
#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_device_utils.cuh"

template<bool REFCTF, bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_wavg(
		XFLOAT *g_eulers,
		CudaProjectorKernel projector,
		unsigned image_size,
		unsigned long orientation_num,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_ctfs,
		XFLOAT *g_wdiff2s_parts,
		XFLOAT *g_wdiff2s_AA,
		XFLOAT *g_wdiff2s_XA,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale)
{
	XFLOAT ref_real, ref_imag, img_real, img_imag, trans_real, trans_imag;

	int bid = blockIdx.x; //block ID
	int tid = threadIdx.x;

	extern __shared__ XFLOAT buffer[];

	unsigned pass_num(ceilfracf(image_size,block_sz)),pixel;
	XFLOAT * s_wdiff2s_parts	= &buffer[0];
	XFLOAT * s_sumXA			= &buffer[block_sz];
	XFLOAT * s_sumA2			= &buffer[2*block_sz];
	XFLOAT * s_eulers           = &buffer[3*block_sz];

	if (tid < 9)
		s_eulers[tid] = g_eulers[bid*9+tid];
	__syncthreads();

	for (unsigned pass = 0; pass < pass_num; pass++) // finish a reference proj in each block
	{
		s_wdiff2s_parts[tid] = 0.0f;
		s_sumXA[tid] = 0.0f;
		s_sumA2[tid] = 0.0f;

		pixel = pass * block_sz + tid;

		if(pixel<image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(pixel, projector.imgX*projector.imgY);
				xy = pixel % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
				{
					if (z >= projector.imgZ - projector.maxR)
						z = z - projector.imgZ;
					else
						x = projector.maxR;
				}
			}
			else
			{
				x =             pixel % projector.imgX;
				y = floorfracf( pixel , projector.imgX);
			}
			if (y > projector.maxR)
			{
				if (y >= projector.imgY - projector.maxR)
					y = y - projector.imgY;
				else
					x = projector.maxR;
			}

			if(DATA3D)
				projector.project3Dmodel(
					x,y,z,
					s_eulers[0], s_eulers[1], s_eulers[2],
					s_eulers[3], s_eulers[4], s_eulers[5],
					s_eulers[6], s_eulers[7], s_eulers[8],
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					s_eulers[6], s_eulers[7],
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
						x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					ref_real, ref_imag);

			if (REFCTF)
			{
				ref_real *= __ldg(&g_ctfs[pixel]);
				ref_imag *= __ldg(&g_ctfs[pixel]);
			}
			else
			{
				ref_real *= part_scale;
				ref_imag *= part_scale;
			}

			img_real = __ldg(&g_img_real[pixel]);
			img_imag = __ldg(&g_img_imag[pixel]);
            //XFLOAT cur_ctf = __ldg(&g_ctfs[pixel]);

			for (unsigned long itrans = 0; itrans < translation_num; itrans++)
			{
				XFLOAT weight = __ldg(&g_weights[bid * translation_num + itrans]);

				if (weight >= significant_weight)
				{
					weight /= weight_norm;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, trans_real, trans_imag);
					else
						translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, trans_real, trans_imag);

					//XFLOAT diff_real = ref_real*cur_ctf - trans_real;
					//XFLOAT diff_imag = ref_imag*cur_ctf - trans_imag;

					//s_wdiff2s_parts[tid] += weight * (diff_real*diff_real + diff_imag*diff_imag);
                    s_wdiff2s_parts[tid] += weight;
					s_sumXA[tid] +=  weight * ( ref_real * trans_real + ref_imag * trans_imag);
					s_sumA2[tid] +=  weight * ( ref_real*ref_real  +  ref_imag*ref_imag );
				}
			}
            s_wdiff2s_parts[tid] *= (img_real*img_real + img_imag*img_imag);

			cuda_atomic_add(&g_wdiff2s_XA[pixel], s_sumXA[tid]);
			cuda_atomic_add(&g_wdiff2s_AA[pixel], s_sumA2[tid]);
			cuda_atomic_add(&g_wdiff2s_parts[pixel], s_wdiff2s_parts[tid]);
		}
	}
}

template<bool refine_ctf, bool save_proj, bool REFCTF, bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_wavg(
		XFLOAT *g_eulers,
		CudaProjectorKernel projector,
		unsigned image_size,
		unsigned long orientation_num,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
        XFLOAT* g_ori_idx,
        XFLOAT* g_ori_proj,
        XFLOAT* g_ori_image,
		XFLOAT* g_ctfs,
		XFLOAT *g_wdiff2s_parts,
		XFLOAT *g_wdiff2s_AA,
		XFLOAT *g_wdiff2s_XA,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale)
{
	XFLOAT ref_real, ref_imag, img_real, img_imag, trans_real, trans_imag;

	int bid = blockIdx.x; //block ID
	int tid = threadIdx.x;

	extern __shared__ XFLOAT buffer[];

	unsigned pass_num(ceilfracf(image_size,block_sz)),pixel;
	XFLOAT * s_wdiff2s_parts	= &buffer[0];
	XFLOAT * s_sumXA			= &buffer[block_sz];
	XFLOAT * s_sumA2			= &buffer[2*block_sz];
	XFLOAT * s_eulers           = &buffer[3*block_sz];

	if (tid < 9)
		s_eulers[tid] = g_eulers[bid*9+tid];
	__syncthreads();

	for (unsigned pass = 0; pass < pass_num; pass++) // finish a reference proj in each block
	{
		s_wdiff2s_parts[tid] = 0.0f;
		s_sumXA[tid] = 0.0f;
		s_sumA2[tid] = 0.0f;

		pixel = pass * block_sz + tid;

		if(pixel<image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(pixel, projector.imgX*projector.imgY);
				xy = pixel % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
				{
					if (z >= projector.imgZ - projector.maxR)
						z = z - projector.imgZ;
					else
						x = projector.maxR;
				}
			}
			else
			{
				x =             pixel % projector.imgX;
				y = floorfracf( pixel , projector.imgX);
			}
			if (y > projector.maxR)
			{
				if (y >= projector.imgY - projector.maxR)
					y = y - projector.imgY;
				else
					x = projector.maxR;
			}

			if(DATA3D)
				projector.project3Dmodel(
					x,y,z,
					s_eulers[0], s_eulers[1], s_eulers[2],
					s_eulers[3], s_eulers[4], s_eulers[5],
					s_eulers[6], s_eulers[7], s_eulers[8],
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					s_eulers[6], s_eulers[7],
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
						x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					ref_real, ref_imag);

			if (REFCTF)
			{
				ref_real *= __ldg(&g_ctfs[pixel]);
				ref_imag *= __ldg(&g_ctfs[pixel]);
			}
			else
			{
				ref_real *= part_scale;
				ref_imag *= part_scale;
			}

			img_real = __ldg(&g_img_real[pixel]);
			img_imag = __ldg(&g_img_imag[pixel]);
            XFLOAT real_diff = 0.;
            XFLOAT imag_diff = 0.;
            XFLOAT tot_weight = 0.;
            XFLOAT t_ctf = __ldg(&g_ctfs[pixel]);
            t_ctf /= part_scale;

			for (unsigned long itrans = 0; itrans < translation_num; itrans++)
			{
				XFLOAT weight = __ldg(&g_weights[bid * translation_num + itrans]);

				if (weight >= significant_weight)
				{
					weight /= weight_norm;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, trans_real, trans_imag);
					else
						translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, trans_real, trans_imag);

					//XFLOAT diff_real = trans_real - ref_real*t_ctf;//ref_real*t_ctf;// - trans_real;
					//XFLOAT diff_imag = trans_imag - ref_imag*t_ctf;//ref_imag*t_ctf;// - trans_imag;

					//s_wdiff2s_parts[tid] += weight * (diff_real*diff_real + diff_imag*diff_imag);
                    real_diff += weight*trans_real;
                    imag_diff += weight*trans_imag;
                    tot_weight += weight;
					s_sumXA[tid] +=  weight * ( ref_real * trans_real + ref_imag * trans_imag);
					s_sumA2[tid] +=  weight * ( ref_real*ref_real  +  ref_imag*ref_imag );
				}
			}
            if(tot_weight == 0.) continue;
            s_wdiff2s_parts[tid] = tot_weight*(img_real*img_real + img_imag*img_imag);

            //orientation index
						if(save_proj)
							{
            int ori_idx = __float2int_rd(g_ori_idx[bid]);
            if(ori_idx >= 0){
                //store projection to ori_idx
							ori_idx += 1;
                g_ori_proj[2*(ori_idx*image_size + pixel)] = ref_real*t_ctf*tot_weight;
                g_ori_proj[2*(ori_idx*image_size + pixel) + 1] = ref_imag*t_ctf*tot_weight;
                g_ori_image[2*(ori_idx*image_size + pixel)] = real_diff;
                g_ori_image[2*(ori_idx*image_size + pixel) + 1] = imag_diff;
            }
							}
						if(refine_ctf)
						{
						cuda_atomic_add(&g_ori_proj[2*pixel], ref_real*tot_weight);
						cuda_atomic_add(&g_ori_proj[2*pixel + 1], ref_imag*tot_weight);
						cuda_atomic_add(&g_ori_image[2*pixel], real_diff);
						cuda_atomic_add(&g_ori_image[2*pixel + 1], imag_diff);
						}
			cuda_atomic_add(&g_wdiff2s_XA[pixel], s_sumXA[tid]);
			cuda_atomic_add(&g_wdiff2s_AA[pixel], s_sumA2[tid]);
			cuda_atomic_add(&g_wdiff2s_parts[pixel], s_wdiff2s_parts[tid]);
		}
	}
}

#endif /* CUDA_WAVG_KERNEL_CUH_ */
