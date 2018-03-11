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

	//opt by ljx prefetch data
	XFLOAT * s_weights			= &buffer[3*block_sz+9];
	XFLOAT * s_trans_x			= &buffer[3*block_sz+9+translation_num];
	XFLOAT * s_trans_y			= &buffer[3*block_sz+9+2*translation_num];
	XFLOAT * s_trans_z			= &buffer[3*block_sz+9+3*translation_num];

	if (tid < 9)
		s_eulers[tid] = g_eulers[bid*9+tid];

	__syncthreads();

	int tran_pass_num(translation_num / block_sz);
	int remain_num(translation_num % block_sz);

	for (unsigned pass = 0; pass < tran_pass_num; pass++)
	{
		s_weights[pass]		= g_weights[bid * translation_num + pass * block_sz + tid];
		s_trans_x[pass]		= g_trans_x[pass * block_sz + tid];
		s_trans_y[pass]		= g_trans_y[pass * block_sz + tid];
		if(DATA3D)
			s_trans_y[pass]	= g_trans_y[pass * block_sz + tid];
	}

	__syncthreads();

	if (tid < remain_num)
	{
		s_weights[tran_pass_num * block_sz + tid]		= g_weights[bid * translation_num + tran_pass_num * block_sz + tid];
		s_trans_x[tran_pass_num * block_sz + tid]		= g_trans_x[tran_pass_num * block_sz + tid];
		s_trans_y[tran_pass_num * block_sz + tid]		= g_trans_y[tran_pass_num * block_sz + tid];
		if(DATA3D)
			s_trans_z[tran_pass_num * block_sz + tid]	= g_trans_z[tran_pass_num * block_sz + tid];
	}
	__syncthreads();

	for (unsigned pass = 0; pass < pass_num; pass++) // finish a reference proj in each block
	{
		s_wdiff2s_parts[tid] = 0.0f;
		s_sumXA[tid] = 0.0f;
		s_sumA2[tid] = 0.0f;

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

				s_wdiff2s_parts[tid] += weight * (diff_real*diff_real + diff_imag*diff_imag);

				s_sumXA[tid] +=  weight * ( ref_real * trans_real + ref_imag * trans_imag);
				s_sumA2[tid] +=  weight * ( ref_real*ref_real  +  ref_imag*ref_imag );
			}
		}

		cuda_atomic_add(&g_wdiff2s_XA[pixel], s_sumXA[tid]);
		cuda_atomic_add(&g_wdiff2s_AA[pixel], s_sumA2[tid]);
		cuda_atomic_add(&g_wdiff2s_parts[pixel], s_wdiff2s_parts[tid]);
	}
}

#endif /* CUDA_WAVG_KERNEL_CUH_ */
