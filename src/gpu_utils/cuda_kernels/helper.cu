#include "src/gpu_utils/cuda_device_utils.cuh"
#include "src/gpu_utils/cuda_kernels/helper.cuh"
#include "src/gpu_utils/cuda_settings.h"

/*
 * This draft of a kernel assumes input that has jobs which have a single orientation and sequential translations within each job.
 *
 */
__global__ void cuda_kernel_exponentiate_weights_fine(
		XFLOAT *g_pdf_orientation,
		XFLOAT *g_pdf_offset,
		XFLOAT *g_weights,
		XFLOAT avg_diff2,
		int oversamples_orient,
		int oversamples_trans,
		unsigned long *d_rot_id,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num,
		long int job_num)
{
	__shared__ XFLOAT s_weights[SUMW_BLOCK_SIZE];

	// blockid
	int bid  = blockIdx.x;
	//threadid
	int tid = threadIdx.x;

	long int jobid = bid*SUMW_BLOCK_SIZE+tid;

	if (jobid<job_num)
	{
		long int pos = d_job_idx[jobid];
		// index of comparison
		long int ix =  d_rot_id[   pos];   // each thread gets its own orient...
		long int iy = d_trans_idx[ pos];   // ...and it's starting trans...
		long int in =  d_job_num[jobid];    // ...AND the number of translations to go through

		int c_itrans;//, iorient = bid*SUM_BLOCK_SIZE+tid; //, f_itrans;

		// Bacause the partion of work is so arbitrarily divided in this kernel,
		// we need to do some brute idex work to get the correct indices.
		for (int itrans=0; itrans < in; itrans++, iy++)
		{
			c_itrans = ( iy - (iy % oversamples_trans))/ oversamples_trans; //floor(x/y) == (x-(x%y))/y  but less sensitive to x>>y and finite precision
//			f_itrans = iy % oversamples_trans;

			XFLOAT prior = g_pdf_orientation[ix] * g_pdf_offset[c_itrans];          	// Same      for all threads - TODO: should be done once for all trans through warp-parallel execution
			XFLOAT diff2 = g_weights[pos+itrans] - avg_diff2;								// Different for all threads
			// next line because of numerical precision of exp-function
	#if defined(CUDA_DOUBLE_PRECISION)
				if (diff2 > 700.)
					s_weights[tid] = 0.;
				else
					s_weights[tid] = prior * exp(-diff2);
	#else
				if (diff2 > 86.)
					s_weights[tid] = 0.f;
				else
					s_weights[tid] = prior * expf(-diff2);
	#endif
				// TODO: use tabulated exp function? / Sjors  TODO: exp, expf, or __exp in CUDA? /Bjorn
			// Store the weight
			g_weights[pos+itrans] = s_weights[tid]; // TODO put in shared mem
		}
	}
}

__global__ void cuda_kernel_softMaskOutsideMap(	XFLOAT *vol,
												long int vol_size,
												long int xdim,
												long int ydim,
												long int zdim,
												long int xinit,
												long int yinit,
												long int zinit,
												bool do_Mnoise,
												XFLOAT radius,
												XFLOAT radius_p,
												XFLOAT cosine_width	)
{

		int tid = threadIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
		XFLOAT r, raisedcos;

		__shared__ XFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT    partial_sum[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT partial_sum_bg[SOFTMASK_BLOCK_SIZE];

		XFLOAT sum_bg_total =  (XFLOAT)0.0;

		long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE);
		int texel = tid;

		partial_sum[tid]=(XFLOAT)0.0;
		partial_sum_bg[tid]=(XFLOAT)0.0;
		if (do_Mnoise)
		{
			for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
			{
				XFLOAT x,y,z;
				if(texel<vol_size)
				{
					img_pixels[tid]=__ldg(&vol[texel]);

					z = floor( (float) texel                   / (float)((xdim)*(ydim)));
					y = floor( (XFLOAT)(texel-z*(xdim)*(ydim)) / (XFLOAT) xdim );
					x = texel - z*(xdim)*(ydim) - y*xdim;

					z-=zinit;
					y-=yinit;
					x-=xinit;

					r = sqrt(x*x + y*y + z*z);

					if (r < radius)
						continue;
					else if (r > radius_p)
					{
						partial_sum[tid]    += (XFLOAT)1.0;
						partial_sum_bg[tid] += img_pixels[tid];
					}
					else
					{
#if defined(CUDA_DOUBLE_PRECISION)
						raisedcos = 0.5 + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
						raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
						partial_sum[tid] += raisedcos;
						partial_sum_bg[tid] += raisedcos * img_pixels[tid];
					}
				}
			}
		}

		__syncthreads();
		for(int j=(SOFTMASK_BLOCK_SIZE/2); j>0; j/=2)
		{
			if(tid<j)
			{
				partial_sum[tid] += partial_sum[tid+j];
				partial_sum_bg[tid] += partial_sum_bg[tid+j];
			}
			__syncthreads();
		}

		sum_bg_total  = partial_sum_bg[0] / partial_sum[0];


		texel = tid;
		for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
		{
			XFLOAT x,y,z;
			if(texel<vol_size)
			{
				img_pixels[tid]=__ldg(&vol[texel]);

				z =  floor( (float) texel                  / (float)((xdim)*(ydim)));
				y = floor( (XFLOAT)(texel-z*(xdim)*(ydim)) / (XFLOAT)  xdim         );
				x = texel - z*(xdim)*(ydim) - y*xdim;

				z-=zinit;
				y-=yinit;
				x-=xinit;

				r = sqrt(x*x + y*y + z*z);

				if (r < radius)
					continue;
				else if (r > radius_p)
					img_pixels[tid]=sum_bg_total;
				else
				{
#if defined(CUDA_DOUBLE_PRECISION)
					raisedcos = 0.5  + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
					raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
					img_pixels[tid]= img_pixels[tid]*(1-raisedcos) + sum_bg_total*raisedcos;

				}
				vol[texel]=img_pixels[tid];
			}

		}
}

__global__ void cuda_kernel_softMaskBackgroundValue(	XFLOAT *vol,
														long int vol_size,
														long int xdim,
														long int ydim,
														long int zdim,
														long int xinit,
														long int yinit,
														long int zinit,
														bool do_Mnoise,
														XFLOAT radius,
														XFLOAT radius_p,
														XFLOAT cosine_width,
														XFLOAT *g_sum,
														XFLOAT *g_sum_bg)
{

		int tid = threadIdx.x;
		int bid = blockIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
		XFLOAT r, raisedcos;
		int x,y,z;
		__shared__ XFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT    partial_sum[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT partial_sum_bg[SOFTMASK_BLOCK_SIZE];

		long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE*gridDim.x);
		int texel = bid*SOFTMASK_BLOCK_SIZE*texel_pass_num + tid;

		partial_sum[tid]=(XFLOAT)0.0;
		partial_sum_bg[tid]=(XFLOAT)0.0;

		for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
		{
			if(texel<vol_size)
			{
				img_pixels[tid]=__ldg(&vol[texel]);

				z =   texel / (xdim*ydim) ;
				y = ( texel % (xdim*ydim) ) / xdim ;
				x = ( texel % (xdim*ydim) ) % xdim ;

				z-=zinit;
				y-=yinit;
				x-=xinit;

				r = sqrt(XFLOAT(x*x + y*y + z*z));

				if (r < radius)
					continue;
				else if (r > radius_p)
				{
					partial_sum[tid]    += (XFLOAT)1.0;
					partial_sum_bg[tid] += img_pixels[tid];
				}
				else
				{
#if defined(CUDA_DOUBLE_PRECISION)
					raisedcos = 0.5 + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
					raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
					partial_sum[tid] += raisedcos;
					partial_sum_bg[tid] += raisedcos * img_pixels[tid];
				}
			}
		}

		cuda_atomic_add(&g_sum[tid]   , partial_sum[tid]);
		cuda_atomic_add(&g_sum_bg[tid], partial_sum_bg[tid]);
}


__global__ void cuda_kernel_cosineFilter(	XFLOAT *vol,
											long int vol_size,
											long int xdim,
											long int ydim,
											long int zdim,
											long int xinit,
											long int yinit,
											long int zinit,
											bool do_Mnoise,
											XFLOAT radius,
											XFLOAT radius_p,
											XFLOAT cosine_width,
											XFLOAT bg_value)
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
	XFLOAT r, raisedcos;
	int x,y,z;
	__shared__ XFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];

	long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE*gridDim.x);
	int texel = bid*SOFTMASK_BLOCK_SIZE*texel_pass_num + tid;

	for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
	{
		if(texel<vol_size)
		{
			img_pixels[tid]=__ldg(&vol[texel]);

			z =   texel / (xdim*ydim) ;
			y = ( texel % (xdim*ydim) ) / xdim ;
			x = ( texel % (xdim*ydim) ) % xdim ;

			z-=zinit;
			y-=yinit;
			x-=xinit;

			r = sqrt(XFLOAT(x*x + y*y + z*z));

			if (r < radius)
				continue;
			else if (r > radius_p)
				img_pixels[tid]=bg_value;
			else
			{
#if defined(CUDA_DOUBLE_PRECISION)
				raisedcos = 0.5  + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
				raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
				img_pixels[tid]= img_pixels[tid]*(1-raisedcos) + bg_value*raisedcos;

			}
			vol[texel]=img_pixels[tid];
		}

	}
}


__global__ void cuda_kernel_translate2D(	XFLOAT * g_image_in,
											XFLOAT * g_image_out,
											int image_size,
											int xdim,
											int ydim,
											int dx,
											int dy)
{
	int tid = threadIdx.x;
	int bid =  blockIdx.x;

	int x,y,xp,yp;
	int pixel=tid + bid*BLOCK_SIZE;
	int new_pixel;

	if(pixel<image_size)
	{
		x = pixel % xdim;
		y = (pixel-x) / (xdim);

		xp = x + dx;
		yp = y + dy;

		if( yp>=0 && xp>=0 && yp<ydim && xp<xdim)
		{
			new_pixel = yp*xdim + xp;
			if(new_pixel>=0 && new_pixel<image_size) // if displacement is negative, new_pixel could be less than 0
				g_image_out[new_pixel] = g_image_in[pixel];
		}
	}
}

__global__ void cuda_kernel_translate3D(	XFLOAT * g_image_in,
											XFLOAT * g_image_out,
											int image_size,
											int xdim,
											int ydim,
											int zdim,
											int dx,
											int dy,
											int dz)
{
	int tid = threadIdx.x;
	int bid =  blockIdx.x;

	int x,y,z,xp,yp,zp,xy;
	int voxel=tid + bid*BLOCK_SIZE;
	int new_voxel;

	int xydim = xdim*ydim;

	if(voxel<image_size)
	{
		z =  voxel / xydim;
		zp = z + dz;

		xy = voxel % xydim;
		y =  xy / xdim;
		yp = y + dy;

		x =  xy % xdim;
		xp = x + dx;

		if( zp>=0 && yp>=0 && xp>=0 && zp<zdim && yp<ydim && xp<xdim)
		{
			new_voxel = zp*xydim +  yp*xdim + xp;
			if(new_voxel>=0 && new_voxel<image_size) // if displacement is negative, new_pixel could be less than 0
				g_image_out[new_voxel] = g_image_in[voxel];
		}
	}
}

__global__ void cuda_kernel_centerFFT_2D(XFLOAT *img_in,
										 int image_size,
										 int xdim,
										 int ydim,
										 int xshift,
										 int yshift)
{

	__shared__ XFLOAT buffer[CFTT_BLOCK_SIZE];
	int tid = threadIdx.x;
	int pixel = threadIdx.x + blockIdx.x*CFTT_BLOCK_SIZE;
	long int image_offset = image_size*blockIdx.y;
//	int pixel_pass_num = ceilfracf(image_size, CFTT_BLOCK_SIZE);

//	for (int pass = 0; pass < pixel_pass_num; pass++, pixel+=CFTT_BLOCK_SIZE)
//	{
		if(pixel<(image_size/2))
		{
			int y = floorf((XFLOAT)pixel/(XFLOAT)xdim);
			int x = pixel % xdim;				// also = pixel - y*xdim, but this depends on y having been calculated, i.e. serial evaluation

			int yp = y + yshift;
			if (yp < 0)
				yp += ydim;
			else if (yp >= ydim)
				yp -= ydim;

			int xp = x + xshift;
			if (xp < 0)
				xp += xdim;
			else if (xp >= xdim)
				xp -= xdim;

			int n_pixel = yp*xdim + xp;

			buffer[tid]                    = img_in[image_offset + n_pixel];
			img_in[image_offset + n_pixel] = img_in[image_offset + pixel];
			img_in[image_offset + pixel]   = buffer[tid];
		}
//	}
}

__global__ void cuda_kernel_centerFFT_3D(XFLOAT *img_in,
										 int image_size,
										 int xdim,
										 int ydim,
										 int zdim,
										 int xshift,
										 int yshift,
									 	 int zshift)
{

	__shared__ XFLOAT buffer[CFTT_BLOCK_SIZE];
	int tid = threadIdx.x;
	int pixel = threadIdx.x + blockIdx.x*CFTT_BLOCK_SIZE;
	long int image_offset = image_size*blockIdx.y;

		int xydim = xdim*ydim;
		if(pixel<(image_size/2))
		{
			int z = floorf((XFLOAT)pixel/(XFLOAT)(xydim));
			int xy = pixel % xydim;
			int y = floorf((XFLOAT)xy/(XFLOAT)xdim);
			int x = xy % xdim;


			int yp = y + yshift;
			if (yp < 0)
				yp += ydim;
			else if (yp >= ydim)
				yp -= ydim;

			int xp = x + xshift;
			if (xp < 0)
				xp += xdim;
			else if (xp >= xdim)
				xp -= xdim;

			int zp = z + zshift;
			if (zp < 0)
				zp += zdim;
			else if (zp >= zdim)
				zp -= zdim;

			int n_pixel = zp*xydim + yp*xdim + xp;

			buffer[tid]                    = img_in[image_offset + n_pixel];
			img_in[image_offset + n_pixel] = img_in[image_offset + pixel];
			img_in[image_offset + pixel]   = buffer[tid];
		}
}


__global__ void cuda_kernel_probRatio(  XFLOAT *d_Mccf,
										XFLOAT *d_Mpsi,
										XFLOAT *d_Maux,
										XFLOAT *d_Mmean,
										XFLOAT *d_Mstddev,
										int image_size,
										XFLOAT normfft,
										XFLOAT sum_ref_under_circ_mask,
										XFLOAT sum_ref2_under_circ_mask,
										XFLOAT expected_Pratio,
										int NpsiThisBatch,
										int startPsi,
										int totalPsis)
{
	/* PLAN TO:
	 *
	 * 1) Pre-filter
	 * 		d_Mstddev[i] = 1 / (2*d_Mstddev[i])   ( if d_Mstddev[pixel] > 1E-10 )
	 * 		d_Mstddev[i] = 1    				  ( else )
	 *
	 * 2) Set
	 * 		sum_ref2_under_circ_mask /= 2.
	 *
	 * 3) Total expression becomes
	 * 		diff2 = ( exp(k) - 1.f ) / (expected_Pratio - 1.f)
	 * 	  where
	 * 	  	k = (normfft * d_Maux[pixel] + d_Mmean[pixel] * sum_ref_under_circ_mask)*d_Mstddev[i] + sum_ref2_under_circ_mask
	 *
	 */

	int pixel = threadIdx.x + blockIdx.x*(int)PROBRATIO_BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT Kccf = d_Mccf[pixel];
		XFLOAT Kpsi =(XFLOAT)-1.0;
		for(int psi = 0; psi < NpsiThisBatch; psi++ )
		{
			XFLOAT diff2 = normfft * d_Maux[pixel + image_size*psi];
			diff2 += d_Mmean[pixel] * sum_ref_under_circ_mask;

	//		if (d_Mstddev[pixel] > (XFLOAT)1E-10)
			diff2 *= d_Mstddev[pixel];
			diff2 += sum_ref2_under_circ_mask;

#if defined(CUDA_DOUBLE_PRECISION)
			diff2 = exp(-diff2 / 2.); // exponentiate to reflect the Gaussian error model. sigma=1 after normalization, 0.4=1/sqrt(2pi)
#else
			diff2 = expf(-diff2 / 2.f);
#endif

			// Store fraction of (1 - probability-ratio) wrt  (1 - expected Pratio)
			diff2 = (diff2 - (XFLOAT)1.0) / (expected_Pratio - (XFLOAT)1.0);
			if (diff2 > Kccf)
			{
				Kccf = diff2;
				Kpsi = (startPsi + psi)*(360/totalPsis);
			}
		}
		d_Mccf[pixel] = Kccf;
		if (Kpsi >= 0.)
			d_Mpsi[pixel] = Kpsi;
	}
}

__global__ void cuda_kernel_rotateOnly(   CUDACOMPLEX *d_Faux,
						  	  	  	  	  XFLOAT psi,
						  	  			  CudaProjectorKernel projector,
						  	  			  int startPsi
						  	  			  )
{
	int proj = blockIdx.y;
	int image_size=projector.imgX*projector.imgY;
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		int y = floorfracf(pixel,projector.imgX);
		int x = pixel % projector.imgX;

		if (y > projector.maxR)
		{
			if (y >= projector.imgY - projector.maxR)
				y = y - projector.imgY;
			else
				x = projector.maxR;
		}

		XFLOAT sa, ca;
		sincos((proj+startPsi)*psi, &sa, &ca);
		CUDACOMPLEX val;

		projector.project2Dmodel(	 x,y,
									 ca,
									-sa,
									 sa,
									 ca,
									 val.x,val.y);

		long int out_pixel = proj*image_size + pixel;

		d_Faux[out_pixel].x =val.x;
		d_Faux[out_pixel].y =val.y;
	}
}

__global__ void cuda_kernel_rotateAndCtf( CUDACOMPLEX *d_Faux,
						  	  	  	  	  XFLOAT *d_ctf,
						  	  	  	  	  XFLOAT psi,
						  	  			  CudaProjectorKernel projector,
						  	  			  int startPsi
						  	  			  )
{
	int proj = blockIdx.y;
	int image_size=projector.imgX*projector.imgY;
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		int y = floorfracf(pixel,projector.imgX);
		int x = pixel % projector.imgX;

		if (y > projector.maxR)
		{
			if (y >= projector.imgY - projector.maxR)
				y = y - projector.imgY;
			else
				x = projector.maxR;
		}

		XFLOAT sa, ca;
		sincos((proj+startPsi)*psi, &sa, &ca);
		CUDACOMPLEX val;

		projector.project2Dmodel(	 x,y,
									 ca,
									-sa,
									 sa,
									 ca,
									 val.x,val.y);

		long int out_pixel = proj*image_size + pixel;

		d_Faux[out_pixel].x =val.x*d_ctf[pixel];
		d_Faux[out_pixel].y =val.y*d_ctf[pixel];

	}
}


__global__ void cuda_kernel_convol_A( CUDACOMPLEX *d_A,
									 CUDACOMPLEX *d_B,
									 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel].x;
		XFLOAT ti = - d_A[pixel].y;
		d_A[pixel].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_A[pixel].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_convol_A( CUDACOMPLEX *d_A,
									 CUDACOMPLEX *d_B,
									 CUDACOMPLEX *d_C,
									 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel].x;
		XFLOAT ti = - d_A[pixel].y;
		d_C[pixel].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_C[pixel].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_batch_convol_A( CUDACOMPLEX *d_A,
									 	 	CUDACOMPLEX *d_B,
									 	 	int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	int A_off = blockIdx.y*image_size;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel + A_off].x;
		XFLOAT ti = - d_A[pixel + A_off].y;
		d_A[pixel + A_off].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_A[pixel + A_off].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_batch_convol_A( CUDACOMPLEX *d_A,
									 	 	CUDACOMPLEX *d_B,
									 	 	CUDACOMPLEX *d_C,
									 	 	int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	int A_off = blockIdx.y*image_size;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel + A_off].x;
		XFLOAT ti = - d_A[pixel + A_off].y;
		d_C[pixel + A_off].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_C[pixel + A_off].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_convol_B(	 CUDACOMPLEX *d_A,
									 	 CUDACOMPLEX *d_B,
									 	 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr = d_A[pixel].x;
		XFLOAT ti = d_A[pixel].y;
		d_A[pixel].x =   tr*d_B[pixel].x + ti*d_B[pixel].y;
		d_A[pixel].y =   ti*d_B[pixel].x - tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_convol_B(	 CUDACOMPLEX *d_A,
									 	 CUDACOMPLEX *d_B,
									 	 CUDACOMPLEX *d_C,
									 	 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr = d_A[pixel].x;
		XFLOAT ti = d_A[pixel].y;
		d_C[pixel].x =   tr*d_B[pixel].x + ti*d_B[pixel].y;
		d_C[pixel].y =   ti*d_B[pixel].x - tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_batch_convol_B(	 CUDACOMPLEX *d_A,
									 	 	 CUDACOMPLEX *d_B,
									 	 	 int image_size)
{
	long int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	int A_off = blockIdx.y*image_size;
	if(pixel<image_size)
	{
		XFLOAT tr = d_A[pixel + A_off].x;
		XFLOAT ti = d_A[pixel + A_off].y;
		d_A[pixel + A_off].x =   tr*d_B[pixel].x + ti*d_B[pixel].y;
		d_A[pixel + A_off].y =   ti*d_B[pixel].x - tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_multi( XFLOAT *A,
								   XFLOAT *OUT,
								   XFLOAT S,
		  	  	  	  	  	  	   int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		OUT[pixel] = A[pixel]*S;
}

__global__ void cuda_kernel_multi(
		XFLOAT *A,
		XFLOAT S,
		int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		A[pixel] = A[pixel]*S;
}

__global__ void cuda_kernel_multi( XFLOAT *A,
								   XFLOAT *B,
								   XFLOAT *OUT,
								   XFLOAT S,
		  	  	  	  	  	  	   int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size){
		OUT[pixel] = A[pixel]*B[pixel]*S;
    }
}

__global__ void cuda_kernel_complex_multi( XFLOAT *A,
                                   XFLOAT *B,
                                   XFLOAT S,
                                   int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        A[pixel*2] *= B[pixel]*S;
        A[pixel*2+1] *= B[pixel]*S;
    }
}

__global__ void cuda_kernel_complex_multi( XFLOAT *A,
                                   XFLOAT *B,
                                   XFLOAT S,
                                   XFLOAT w,
                                   int Z,
                                   int Y,
                                   int X,
                                   int ZZ,
                                   int YY,
                                   int XX,
                                   int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        int kp = pixel / (Y*X);
        int ip = (pixel - kp * (Y*X))/X;
        int jp = pixel - kp * (Y*X) - ip * X;
        if(kp >= X) kp -= (Z);
        if(ip >= X) ip -= (Y);
        if(kp < XX && kp > -XX && ip < XX && ip > -XX && jp < XX) {
            if(kp < 0) kp += ZZ;
            if(ip < 0) ip += YY;
            int n_pixel = kp*(YY*XX) + ip*XX + jp;
            A[pixel*2] *= (B[n_pixel]*S + w);
            A[pixel*2+1] *= (B[n_pixel]*S + w);
        } else {
            //A[pixel*2] = 0.;
            //A[pixel*2+1] = 0.;
            A[pixel*2] *=w;
            A[pixel*2+1] *=w;
        }
    }
}

__global__ void cuda_kernel_batch_multi( XFLOAT *A,
								   XFLOAT *B,
								   XFLOAT *OUT,
								   XFLOAT S,
		  	  	  	  	  	  	   int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		OUT[pixel + blockIdx.y*image_size] = A[pixel + blockIdx.y*image_size]*B[pixel + blockIdx.y*image_size]*S;
}

__global__ void cuda_kernel_substract(XFLOAT *A,
                                     XFLOAT *B,
                                     int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        A[pixel] -= B[pixel];
    }
}

__global__ void cuda_kernel_substract(XFLOAT *A,
                                     XFLOAT *B,
                                     XFLOAT *C,
                                     XFLOAT l,
                                     int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        A[pixel] -= (B[pixel] - l*C[pixel]);
    }
}

__global__ void cuda_kernel_substract(XFLOAT *A,
                                     XFLOAT *B,
                                     XFLOAT *C,
                                     XFLOAT l,
                                     int Z,
                                     int Y,
                                     int X,
                                     int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        int kp = pixel / (Y*X);
        int ip = (pixel - kp * (Y*X))/X;
        int jp = pixel - kp * (Y*X) - ip * X;
        int hZ = Z >> 1;
        int hY = Y >> 1;
        int hX = X >> 1;
        if(kp >= hZ) kp += Z;
        if(ip >= hY) ip += Y;
        if(jp >= hX) jp += X;
        hY = Y << 1;
        hX = X << 1;
        int c_pixel = kp*hY*hX + ip*hX + jp;
        A[c_pixel] -= (B[c_pixel] - l*C[c_pixel]);
    }
}

__global__ void cuda_kernel_substract(XFLOAT *A,
                                     XFLOAT *B,
                                     XFLOAT *C,
                                     XFLOAT *vol_out,
                                     XFLOAT l,
                                     XFLOAT* sum,
                                     int Z,
                                     int Y,
                                     int X,
                                     int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        int kp = pixel / (Y*X);
        int ip = (pixel - kp * (Y*X))/X;
        int jp = pixel - kp * (Y*X) - ip * X;
        int hZ = Z >> 1;
        int hY = Y >> 1;
        int hX = X >> 1;
        if(kp >= hZ) kp += Z;
        if(ip >= hY) ip += Y;
        if(jp >= hX) jp += X;
        hY = Y << 1;
        hX = X << 1;
        int c_pixel = kp*hY*hX + ip*hX + jp;
        XFLOAT tmp = B[c_pixel] - vol_out[c_pixel];
        tmp -= A[c_pixel];
        A[c_pixel] -= (B[c_pixel] - l*C[c_pixel]);
        cuda_atomic_add(&sum[0], tmp*tmp);
    }
}
__global__ void cuda_kernel_update_momentum(XFLOAT *grads,
                                            XFLOAT *momentum,
                                            XFLOAT mu,
                                            XFLOAT l_r,
                                            int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        XFLOAT tmp = momentum[pixel];
        momentum[pixel] = mu*momentum[pixel] - l_r*grads[pixel];
        grads[pixel] = tmp;
    }
}

__global__ void cuda_kernel_soft_threshold(XFLOAT *img,
                                           XFLOAT *momentum,
                                           XFLOAT *grads,
                                           int Z,
                                           int Y,
                                           int X,
                                           XFLOAT mu,
                                           XFLOAT l_r,
                                           XFLOAT alpha,
                                           XFLOAT eps,
                                           int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        int kp = pixel / (Y*X);
        int ip = (pixel - kp * (Y*X))/X;
        int jp = pixel - kp*(Y*X) - ip*X;
        int hZ = Z >> 1;
        int hY = Y >> 1;
        int hX = X >> 1;
        if(kp >= hZ) kp += Z;
        if(ip >= hY) ip += Y;
        if(jp >= hX) jp += X;
        hY = Y << 1;
        hX = X << 1;
        int c_pixel = kp*hY*hX + ip*hX + jp;
        XFLOAT th = l_r*alpha/(eps+img[c_pixel]);
        XFLOAT tmp = momentum[pixel];
        XFLOAT n_momentum = mu*tmp - l_r*grads[c_pixel];
        momentum[pixel] = n_momentum;
        grads[c_pixel] = img[c_pixel];
        img[c_pixel] += n_momentum + mu*(n_momentum - tmp);
        if(img[c_pixel] < th && img[c_pixel] > -th){
            img[c_pixel] = 0.;
        } else {
            if(img[c_pixel] >= th){
                img[c_pixel] -= th;
            } else {
                img[c_pixel] += th;
            }
        }
        grads[c_pixel] -= img[c_pixel];
    }
}

__global__ void cuda_kernel_soft_threshold(XFLOAT *img,
                                           XFLOAT *grads,
                                           XFLOAT l_r,
                                           XFLOAT alpha,
                                           XFLOAT eps,
                                           int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        XFLOAT th = l_r*alpha/(eps+img[pixel]);
        XFLOAT tmp = img[pixel];
        img[pixel] -=  l_r*grads[pixel];
        grads[pixel] = tmp;
        if(img[pixel] < th && img[pixel] > -th){
            img[pixel] = 0.;
        } else {
            if(img[pixel] >= th){
                img[pixel] -= th;
            } else {
                img[pixel] += th;
            }
        }
        grads[pixel] -= img[pixel];
    }
}

__global__ void cuda_kernel_soft_threshold(XFLOAT *img,
                                           XFLOAT *grads,
                                           XFLOAT l_r,
                                           XFLOAT alpha,
                                           XFLOAT eps,
                                           int X,
                                           int Y,
                                           int Z,
                                           int XX,
                                           int YY,
                                           int ZZ,
                                           int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        int k = pixel/(YY*XX);
        int i = (pixel - k*YY*XX)/XX;
        int j = pixel - k*YY*XX - i*XX;
        int hZ = ZZ >> 1;
        int hY = YY >> 1;
        int hX = XX >> 1;
        int kl = k;
        int il = i;
        int jl = j;
        if(kl >= hZ){
            kl -= ZZ;
            k = kl + Z;
        }
        if(il >= hY){
            il -= YY;
            i = il + Y;
        }
        if(jl >= hX){
            jl -= XX;
            j = jl + X;
        }

        pixel = k*Y*X + i*X + j;
        XFLOAT th = l_r*alpha/(eps+img[pixel]);
        XFLOAT tmp = img[pixel];
        img[pixel] -=  l_r*grads[pixel];
        //grads[pixel] = tmp;
        if(img[pixel] < th && img[pixel] > -th){
            img[pixel] = 0.;
        } else {
            if(img[pixel] >= th){
                img[pixel] -= th;
            } else {
                img[pixel] += th;
            }
        }
        //grads[pixel] -= img[pixel];
    }
}

__global__ void cuda_kernel_soft_threshold(XFLOAT *img,
                                           XFLOAT *momentum,
                                           XFLOAT *grads,
                                           XFLOAT mu,
                                           XFLOAT l_r,
                                           XFLOAT alpha,
                                           XFLOAT eps,
                                           int X,
                                           int Y,
                                           int Z,
                                           int XX,
                                           int YY,
                                           int ZZ,
                                           int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        int k = pixel/(YY*XX);
        int i = (pixel - k*YY*XX)/XX;
        int j = pixel - k*YY*XX - i*XX;
        int hZ = ZZ >> 1;
        int hY = YY >> 1;
        int hX = XX >> 1;
        int kl = k;
        int il = i;
        int jl = j;
        if(kl >= hZ){
            kl -= ZZ;
            k = kl + Z;
        }
        if(il >= hY){
            il -= YY;
            i = il + Y;
        }
        if(jl >= hX){
            jl -= XX;
            j = jl + X;
        }
        pixel = k*Y*X + i*X + j;
        XFLOAT th = l_r*alpha/(eps+img[pixel]);
        //store image first
        XFLOAT tmp = img[pixel];
        //threshold result goest to image
        momentum[pixel] -=  l_r*grads[pixel];
        //grads[pixel] = tmp;
        if(momentum[pixel] < th && momentum[pixel] > -th){
            img[pixel] = 0.;
        } else {
            if(momentum[pixel] >= th){
                img[pixel] = momentum[pixel] - th;
            } else {
                img[pixel] = momentum[pixel] + th;
            }
        }
        //mix new image with old image to get new momentum
        grads[pixel] = img[pixel] - tmp;
        momentum[pixel] = img[pixel] + mu*grads[pixel];
    }
}

__global__ void cuda_kernel_soft_threshold(XFLOAT *img,
                                           XFLOAT *momentum,
                                           XFLOAT *grads,
                                           XFLOAT *curvature,
                                           XFLOAT mu,
                                           XFLOAT l_r,
                                           XFLOAT alpha,
                                           XFLOAT eps,
                                           XFLOAT epsadam,
                                           XFLOAT mut,
                                           int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        l_r /= (sqrt(curvature[pixel])+epsadam);
        XFLOAT th = l_r*alpha/(eps+img[pixel]);
        XFLOAT tmp = momentum[pixel];
        XFLOAT tmp_grad = grads[pixel];
        XFLOAT n_momentum = mu*tmp - (1. - mu)*tmp_grad;
        momentum[pixel] = n_momentum;
        grads[pixel] = img[pixel];
        img[pixel] += (mu*n_momentum + (1. - mu)*tmp_grad)*l_r/(1. - mut);
        if(img[pixel] < th && img[pixel] > -th){
            img[pixel] = 0.;
        } else {
            if(img[pixel] >= th){
                img[pixel] -= th;
            } else {
                img[pixel] += th;
            }
        }
        grads[pixel] -= img[pixel];
    }
}

__global__ void cuda_kernel_graph_grad(XFLOAT *img,
                                       XFLOAT *grads,
                                       int Y,
                                       int X,
                                       XFLOAT beta,
                                       XFLOAT eps,
                                       int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        XFLOAT val = img[pixel];
        int i = pixel/X;
        int j = pixel - i*X;
        int hY = Y>>1;
        int hX = X>>1;
        XFLOAT tmp = 0.;
        int il = i;// + hY;
        int jl = j;// + hX;
        if (il >= hY) il -= Y;
        if (jl >= hX) jl -= X;
        //il -= hY;
        //jl -= hX;
        XFLOAT norm = 0.;
        XFLOAT gtmp = 0.;
        if( il < hY - 1){
            int ipp = il + 1;
            if(il < -1) ipp += Y;
            int loc = ipp*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( jl < hX - 1){
            int jpp = jl + 1;
            if(jl < -1) jpp += X;
            int loc = i*X + jpp;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if(norm > eps*eps){
            tmp /= sqrt(norm);
            gtmp += tmp*beta;
        } else {
            gtmp += tmp*beta/eps;
        }
        //got the norm of il - 1, jl
        if( il > -hY ){
            int ipm = il - 1;
            if(il < 1) ipm += Y;
            val = img[ipm*X + j];
            tmp = img[pixel] - val;
            norm = tmp*tmp;
            if( jl < hX - 1){
                int jpp = jl + 1;
                if(jl < -1) jpp += X;
                int loc = ipm*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (val - img_loc)*(val - img_loc);
            }
            if(norm > eps*eps){
                tmp /= sqrt(norm);
                gtmp += tmp*beta;
            } else {
                gtmp += tmp*beta/eps;
            }
        }
        //got the norm of il, jl - 1
        //il ranges from 0, hX - 1, -hX, -1
        if( jl > -hX ){
            int jpm = jl - 1;
            if(jl < 1) jpm += X;
            val = img[i*X + jpm];
            tmp = img[pixel] - val;
            norm = tmp*tmp;
            if( il < hY - 1){
                int ipp = il + 1;
                if(il < -1) ipp += X;
                int loc = ipp*X + jpm;
                XFLOAT img_loc = img[loc];
                norm += (val - img_loc)*(val - img_loc);
            }
            if(norm > eps*eps){
                tmp /= sqrt(norm);
                gtmp += tmp*beta;
            } else {
                gtmp += tmp*beta/eps;
            }
        }
        grads[pixel] += gtmp;
        //if( ip > -hY)
        //{
        //    int ipp = ip - 1;
        //    if(ip < 1) ipp += Y;
        //    int loc = ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( ip < hY - 1)
        //{
        //    int ipp = ip + 1;
        //    if(ip < -1) ipp += Y;
        //    int loc = ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( jp > -hX)
        //{
        //    int jpp = jp - 1;
        //    if(jp < 1) jpp += X;
        //    int loc = i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //if( jp < hX - 1)
        //{
        //    int jpp = jp + 1;
        //    if(jp < -1) jpp += X;
        //    int loc = i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //grads[pixel] += tmp*beta;
    }

}

__global__ void cuda_kernel_graph_grad(XFLOAT *img,
                                       XFLOAT *grads,
                                       int Z,
                                       int Y,
                                       int X,
                                       int ZZ,
                                       int YY,
                                       int XX,
                                       XFLOAT beta,
                                       XFLOAT epslog,
                                       XFLOAT eps,
                                       int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        int k = pixel/(YY*XX);
        int i = (pixel - k*YY*XX)/XX;
        int j = pixel - k*YY*XX - i*XX;
        int hZ = ZZ>>1;
        int hY = YY>>1;
        int hX = XX>>1;
        XFLOAT tmp = 0.;
        int kl = k ;//+ hZ;
        int il = i ;//+ hY;
        int jl = j ;//+ hX;
        if (kl >= hZ) {
            kl -= ZZ;
            k  += ZZ;
        }
        if (il >= hY) {
            il -= YY;
            i  += YY;
        }
        if (jl >= hX) {
            jl -= XX;
            j  += XX;
        }
        XFLOAT val = img[k*Y*X+i*X+j];
        XFLOAT norm = 0.;
        XFLOAT gtmp = 0.;
        int kpp = kl + 1;
        if(kl < -1) kpp += Z;
        int ipp = il + 1;
        if(il < -1) ipp += Y;
        int jpp = jl + 1;
        if(jl < -1) jpp += X;

        if( kl < hZ - 1){
            int loc = kpp*Y*X + i*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( il < hY - 1){
            int loc = k*Y*X + ipp*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( jl < hX - 1){
            int loc = k*Y*X + i*X + jpp;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        norm = sqrt(norm);
        if(norm > eps){
            tmp /= norm;
        } else {
            tmp /= eps;
        }
        gtmp += tmp/(norm + epslog)*beta;
        //got the norm of kl-1, il, jl
        //kl - 1 >= -hZ
        if( kl > -hZ ){
            int kpm = kl - 1;
            //kl - 1 < 0
            if(kl < 1) kpm += Z;
            XFLOAT nval = img[kpm*Y*X + i*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            //il + 1 < hY
            if( il < hY - 1){
                int loc = kpm*Y*X + ipp*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = kpm*Y*X + i*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        //got the norm of kl, il - 1, jl
        if( il > -hY ){
            norm = 0.;
            int ipm = il - 1;
            if(il < 1) ipm += Y;
            XFLOAT nval = img[k*Y*X + ipm*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + ipm*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = k*Y*X + ipm*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp*beta/(norm + epslog);
        }
        //got the norm of kl, il, jl - 1
        if( jl > -hX ){
            int jpm = jl - 1;
            if(jl < 1) jpm += X;
            XFLOAT nval = img[k*Y*X + i*X + jpm];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + i*X + jpm;
                norm += (nval - img[loc])*(nval - img[loc]);
            }
            if( il < hY - 1){
                int loc = k*Y*X + ipp*X + jpm;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        //grads[pixel] += gtmp;
        grads[k*Y*X+i*X+j] += gtmp;
    }
}

/*__global__ void cuda_kernel_graph_grad_shared(XFLOAT *img,
                                       XFLOAT *grads,
                                       int Z,
                                       int Y,
                                       int X,
                                       XFLOAT beta,
                                       XFLOAT epslog,
                                       XFLOAT eps,
                                       int image_size)
{
    __shared__ XFLOAT img_shared[TILE_SIZE][TILE_SIZE][PENCIL_SIZE];
    int iy = threadIdx.y + blockDim.y*blockIdx.y;
    int iz = threadIdx.z + blockDim.z*blockIdx.z;
    if(iy < Y && iz < Z){
        for(int i = 0; i < ceilf(XFLOAT(X)/blockDim.x); i++) {
            int ix = threadIdx.x + i*blockDim.x + blockDim.x*blockIdx.x;
            int pixel = ix + iy*X + iz*Y*X;
            img_shared[threadIdx.z][threadIdx.y][threadIdx.x] = img[pixel];
            if(threadIdx.x == PENCIL_SIZE - 2){
                img_shared[threadIdx.z][threadIdx.y][threadIdx.x+1] = img[pixel + 1];
            }
            if(threadIdx.y == TILE_SIZE - 2) {
                pixel = ix + (iy + 1)*X + iz*Y*X;
                img_shared[threadIdx.z][threadIdx.y+1][threadIdx.x] = img[pixel];
            }
            if(threadIdx.z == TILE_SIZE - 2) {
                pixel = ix + iy*X + (iz + 1)*Y*X;
                img_shared[threadIdx.z+1][threadIdx.y][threadIdx.x] = img[pixel];
            }
            if(threadIdx.x == X - 1){
                pixel = iy*X + iz*Y*X;
                img_shared[threadIdx.z][threadIdx.y][threadIdx.x+1] = img[pixel];
            }
            if(threadIdx.y == Y - 1){
                pixel = ix + iz*Y*X;
                img_shared[threadIdx.z][threadIdx.y+1][threadIdx.x] = img[pixel];
            }
            if(threadIdx.z == Z - 1){
                pixel = ix + iy*X;
                img_shared[threadIdx.z+1][threadIdx.y+1][threadIdx.x] = img[pixel];
            }
            __syncthreads();
            XFLOAT norm = 0.;
            XFLOAT gtmp = 0.;
            XFLOAT diffx = img_shared[threadIdx.z][threadIdx.y][threadIdx.x] - img_shared[threadIdx.z][threadIdx.y][threadIdx.x+1];
            XFLOAT diffy = img_shared[threadIdx.z][threadIdx.y][threadIdx.x] - img_shared[threadIdx.z][threadIdx.y+1][threadIdx.x];
            XFLOAT diffz = img_shared[threadIdx.z][threadIdx.y][threadIdx.x] - img_shared[threadIdx.z+1][threadIdx.y][threadIdx.x];
            norm = diffx*diffx + diffy*diffy + diffz*diffz;
            norm = sqrt(norm);
            gtmp = diffx + diffy + diffz;
            XFLOAT snorm = norm;
            if(norm > eps){
                snorm = eps;
            }
            norm += epslog;
            gtmp = gtmp/snorm/norm*beta;
            pixel = ix + iy*X + iz*Y*X;
            atomicAdd(grad + pixel, gtmp);
            pixel = (ix + 1)%X + iy*X + iz*Y*X;
            gtmp = -diffx/snorm/norm*beta;
            atomicAdd(grad + pixel, gtmp);
            gtmp = -diffy/snorm/norm*beta;
            pixel = ix + ((iy+1) % Y)*X + iz*Y*X;
            atomicAdd(grad + pixel, gtmp);
            gtmp = -diffz/snorm/norm*beta;
            pixel = ix + iy*X + ((iz+1) % Z)*Y*X;
            atomicAdd(grad + pixel, gtmp);
        }

    }

}*/

__global__ void cuda_kernel_graph_grad(XFLOAT *img,
                                       XFLOAT *grads,
                                       int Z,
                                       int Y,
                                       int X,
                                       XFLOAT beta,
                                       XFLOAT epslog,
                                       XFLOAT eps,
                                       int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        XFLOAT val = img[pixel];
        int k = pixel/(Y*X);
        int i = (pixel - k*Y*X)/X;
        int j = pixel - k*Y*X - i*X;
        int hZ = Z>>1;
        int hY = Y>>1;
        int hX = X>>1;
        XFLOAT tmp = 0.;
        int kl = k ;//+ hZ;
        int il = i ;//+ hY;
        int jl = j ;//+ hX;
        if (kl >= hZ) kl -= Z;
        if (il >= hY) il -= Y;
        if (jl >= hX) jl -= X;
        //kl -= hZ;
        //il -= hY;
        //jl -= hX;
        XFLOAT norm = 0.;
        XFLOAT gtmp = 0.;
        int kpp = kl + 1;
        if(kl < -1) kpp += Z;
        int ipp = il + 1;
        if(il < -1) ipp += Y;
        int jpp = jl + 1;
        if(jl < -1) jpp += X;

        if( kl < hZ - 1){
            int loc = kpp*Y*X + i*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( il < hY - 1){
            int loc = k*Y*X + ipp*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( jl < hX - 1){
            int loc = k*Y*X + i*X + jpp;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        norm = sqrt(norm);
        if(norm > eps){
            tmp /= norm;
        } else {
            tmp /= eps;
        }
        gtmp += tmp/(norm + epslog)*beta;
        //got the norm of kl-1, il, jl
        if( kl > -hZ ){
            int kpm = kl - 1;
            if(kl < 1) kpm += Z;
            XFLOAT nval = img[kpm*Y*X + i*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            if( il < hY - 1){
                int loc = kpm*Y*X + ipp*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = kpm*Y*X + i*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        //got the norm of kl, il - 1, jl
        if( il > -hY ){
            norm = 0.;
            int ipm = il - 1;
            if(il < 1) ipm += Y;
            XFLOAT nval = img[k*Y*X + ipm*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + ipm*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = k*Y*X + ipm*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp*beta/(norm + epslog);
        }
        //got the norm of kl, il, jl - 1
        if( jl > -hX ){
            int jpm = jl - 1;
            if(jl < 1) jpm += X;
            XFLOAT nval = img[k*Y*X + i*X + jpm];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + i*X + jpm;
                norm += (nval - img[loc])*(nval - img[loc]);
            }
            if( il < hY - 1){
                int loc = k*Y*X + ipp*X + jpm;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        grads[pixel] += gtmp;
        //if( kp > -hZ)
        //{
        //    int kpp = kp - 1;
        //    if(kp < 1) kpp += Z;
        //    int loc = kpp*Y*X + i*X + j;
        //    tmp += val - img[loc];
        //}
        //if( kp < hZ - 1)
        //{
        //    int kpp = kp + 1;
        //    if(kp < -1) kpp += Z;
        //    int loc = kpp*Y*X + i*X + j;
        //    tmp += val - img[loc];
        //}
        //if( ip > -hY)
        //{
        //    int ipp = ip - 1;
        //    if(ip < 1) ipp += Y;
        //    int loc = k*Y*X + ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( ip < hY - 1)
        //{
        //    int ipp = ip + 1;
        //    if(ip < -1) ipp += Y;
        //    int loc = k*Y*X + ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( jp > -hX)
        //{
        //    int jpp = jp - 1;
        //    if(jp < 1) jpp += X;
        //    int loc = k*Y*X + i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //if( jp < hX - 1)
        //{
        //    int jpp = jp + 1;
        //    if(jp < -1) jpp += X;
        //    int loc = k*Y*X + i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //grads[pixel] += tmp*beta;
    }

}

__global__ void cuda_kernel_finalizeMstddev( XFLOAT *Mstddev,
											 XFLOAT *aux,
											 XFLOAT S,
											 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT temp = Mstddev[pixel] + S * aux[pixel];
		if(temp > 0)
			Mstddev[pixel] = sqrt(temp);
		else
			Mstddev[pixel] = 0;
	}
}

__global__ void cuda_kernel_square(
		XFLOAT *A,
		int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		A[pixel] = A[pixel]*A[pixel];
}

__global__ void cuda_kernel_square(
		XFLOAT *A,
        XFLOAT *B,
        XFLOAT beta,
		int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		B[pixel] = (1. - beta)*A[pixel]*A[pixel] + beta*B[pixel];
}
