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
	XFLOAT wdiff2s_parts;
	XFLOAT sumXA;
	XFLOAT sumA2;

	//opt by ljx prefetch data
	XFLOAT * s_weights			= &buffer[0];                  // max = 21(?) * 4 = 21 B
	XFLOAT * s_trans_x			= &buffer[translation_num];  // max = 21(?) * 4 = 21 B 
	XFLOAT * s_trans_y			= &buffer[2*translation_num];// max = 21(?) * 4 = 21 B 
	XFLOAT * s_trans_z			= &buffer[3*translation_num];// max = 21(?) * 4 = 21 B  

	// now max <= 13KB, 48 - 1 = 47 KB left.

	XFLOAT e0, e1, e2, e3, e4, e5, e6, e7, e8;

	e0 = g_eulers[bid*9+0];
	e1 = g_eulers[bid*9+1];
	e2 = g_eulers[bid*9+2];

	e3 = g_eulers[bid*9+3];
	e4 = g_eulers[bid*9+4];
	e5 = g_eulers[bid*9+5];

	e6 = g_eulers[bid*9+6];
	e7 = g_eulers[bid*9+7];
	e8 = g_eulers[bid*9+8];

	int tran_pass_num(translation_num / block_sz);
	int remain_num(translation_num % block_sz);
	
	if (tran_pass_num != 0)
	{

		for (unsigned pass = 0; pass < tran_pass_num; pass++)
		{
			s_weights[pass * block_sz + tid]		= __ldg(&g_weights[bid * translation_num + pass * block_sz + tid]);
			s_trans_x[pass * block_sz + tid]		= __ldg(&g_trans_x[pass * block_sz + tid]);
			s_trans_y[pass * block_sz + tid]		= __ldg(&g_trans_y[pass * block_sz + tid]);
			if(DATA3D)
				s_trans_z[pass * block_sz + tid]	= __ldg(&g_trans_z[pass * block_sz + tid]);
		}

		__syncthreads();
	}

	if (tid < remain_num)
	{
		s_weights[tran_pass_num * block_sz + tid]		= __ldg(&g_weights[bid * translation_num + tran_pass_num * block_sz + tid]);
		s_trans_x[tran_pass_num * block_sz + tid]		= __ldg(&g_trans_x[tran_pass_num * block_sz + tid]);
		s_trans_y[tran_pass_num * block_sz + tid]		= __ldg(&g_trans_y[tran_pass_num * block_sz + tid]);
		if(DATA3D)
			s_trans_z[tran_pass_num * block_sz + tid]	= __ldg(&g_trans_z[tran_pass_num * block_sz + tid]);
	}
	__syncthreads();

	//XFLOAT sum[pass_num] = {0.0f};

	// pass_num may very large(bigger than 128 in Refine's last iteration, no bigger than 32 in 3D and 64 in 2D), 
	// it means we should use more than 256 * 128 = 32K register per block(with this dataset).......holy shit
	for (unsigned pass = 0; pass < pass_num; pass++) // finish a reference proj in each block
	{
		wdiff2s_parts = 0.0f;
		sumXA = 0.0f;
		sumA2 = 0.0f;

		pixel = pass * block_sz + tid;

		if(pixel >= image_size)
			break;
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
				e0, e1, e2,
				e3, e4, e5,
				e6, e7, e8,
				ref_real, ref_imag);
		else if(REF3D)
			projector.project3Dmodel(
				x,y,
				e0, e1,
				e3, e4,
				e6, e7,
				ref_real, ref_imag);
		else
			projector.project2Dmodel(
					x,y,
				e0, e1,
				e3, e4,
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

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			XFLOAT weight = s_weights[itrans];

			if (weight >= significant_weight)
			{
				weight /= weight_norm;

				if(DATA3D)
					translatePixel(x, y, z, s_trans_x[itrans], s_trans_y[itrans], s_trans_z[itrans], img_real, img_imag, trans_real, trans_imag);
				else
					translatePixel(x, y,    s_trans_x[itrans], s_trans_y[itrans],                    img_real, img_imag, trans_real, trans_imag);

				XFLOAT diff_real = ref_real - trans_real;
				XFLOAT diff_imag = ref_imag - trans_imag;

				wdiff2s_parts += weight * (diff_real*diff_real + diff_imag*diff_imag);

				sumXA +=  weight * ( ref_real * trans_real + ref_imag * trans_imag);
				sumA2 +=  weight * ( ref_real*ref_real  +  ref_imag*ref_imag );
			}
		}

		cuda_atomic_add(&g_wdiff2s_XA[pixel], sumXA);
		cuda_atomic_add(&g_wdiff2s_AA[pixel], sumA2);
		cuda_atomic_add(&g_wdiff2s_parts[pixel], wdiff2s_parts);
	}
}

#endif /* CUDA_WAVG_KERNEL_CUH_ */
