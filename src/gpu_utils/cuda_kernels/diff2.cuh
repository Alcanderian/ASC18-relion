#ifndef CUDA_DIFF2_KERNELS_CUH_
#define CUDA_DIFF2_KERNELS_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "src/gpu_utils/cuda_projector.cuh"
#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_device_utils.cuh"


/*
 *   	DIFFERNECE-BASED KERNELS
 */

/*
 * Assuming block_sz % prefetch_fraction == 0 and prefetch_fraction < block_sz
 * Assuming block_sz % eulers_per_block == 0
 * Assuming eulers_per_block * 3 < block_sz
 */
template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
__global__ void cuda_kernel_diff2_coarse(
		XFLOAT *g_eulers,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		CudaProjectorKernel projector,
		XFLOAT *g_corr,
		XFLOAT *g_diff2s,
		int translation_num,
		int image_size
		)
{
	//int bid = blockIdx.x;
	int tid = threadIdx.x;

	extern __shared__ XFLOAT buffer[];

	//Prefetch euler matrices, this is too big that I can't make it be a register array.
	XFLOAT * s_eulers = &buffer[0]; // size = eulers_per_block * 9 //max = 32 * 9 * 4 = 1152 B

	int max_block_pass_euler( ceilfracf(eulers_per_block*9, block_sz) * block_sz);

	for (int i = tid; i < max_block_pass_euler; i += block_sz)
		if (i < eulers_per_block * 9)
			s_eulers[i] = __ldg(&g_eulers[blockIdx.x * eulers_per_block * 9 + i]);


	//Setup variables, block_sz * eulers_per_block = 2048, 4096, 8192
	XFLOAT * s_ref_real = &buffer[eulers_per_block * 9];                                //[block_sz/prefetch_fraction * eulers_per_block]; // max = 2048 * 8 / 4 * 4 = 16 KB
	XFLOAT * s_ref_imag = &buffer[(9 + block_sz/prefetch_fraction) * eulers_per_block]; //[block_sz/prefetch_fraction * eulers_per_block]; // max = 2048 * 8 / 4 * 4 = 16 KB

	XFLOAT * s_real = &buffer[(9 + ((block_sz/prefetch_fraction) << 1)) * eulers_per_block];//[block_sz]; //max = 128 * 8 * 4 = 4 KB
	XFLOAT * s_imag = &buffer[(9 + ((block_sz/prefetch_fraction) << 1)) * eulers_per_block + block_sz];//[block_sz]; //max = 128 * 8 * 4 = 4 KB
	XFLOAT * s_corr = &buffer[(9 + ((block_sz/prefetch_fraction) << 1)) * eulers_per_block + (block_sz << 1)];//[block_sz]; //max = 128 * 8 * 4 = 4 KB

	//total max = 32 + 12 + 1 = 45 KB, and shared memory per block = 48 KB !!!???

	//use too much shared memory, only 3kb left, but we need MAX = 2048 * 8 * 4 = 64 KB !!!????
	//__shared__ XFLOAT s_diff2s[block_sz * eulers_per_block];
	// so I am going to reuse the shared memory.
	//we can pre caclulate this space we need, if it is more than 48 KB, we use the old method.
	int _48KB = (1 << 10) * 48;
	int _diff_sz = (translation_num * eulers_per_block) * sizeof(XFLOAT);
	XFLOAT * s_diff2s = &buffer[0];//size = translation_num * eulers_per_block; // max size = 130 * 64 * 4 = 32 KB in this case....
	XFLOAT diff2s[eulers_per_block] = {0.f};

	XFLOAT tx = __ldg(&trans_x[tid%translation_num]);
	XFLOAT ty = __ldg(&trans_y[tid%translation_num]);
	XFLOAT tz = __ldg(&trans_z[tid%translation_num]);

	//Step through data
	int max_block_pass_pixel( ceilfracf(image_size,block_sz) * block_sz );

	for (int init_pixel = 0; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
	{
		__syncthreads();

		//Prefetch block-fraction-wise
		if(init_pixel + tid/prefetch_fraction < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(init_pixel + tid/prefetch_fraction, projector.imgX*projector.imgY);
				xy = (init_pixel + tid/prefetch_fraction) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + tid/prefetch_fraction) % projector.imgX;
				y = floorfracf( init_pixel + tid/prefetch_fraction  , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

//			#pragma unroll
			for (int i = tid%prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
			{
				if(DATA3D) // if DATA3D, then REF3D as well.
					projector.project3Dmodel(
						x,y,z,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+2],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+5],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_eulers[i*9+8],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else
					projector.project2Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
			}
		}

		//Prefetch block-wise
		if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
		{
			s_real[tid] = __ldg(&g_real[init_pixel + tid]);
			s_imag[tid] = __ldg(&g_imag[init_pixel + tid]);
			s_corr[tid] = __ldg(&g_corr[init_pixel + tid]) / 2;
		}

		__syncthreads();

		if (tid/translation_num < block_sz/translation_num) // NOTE int division A/B==C/B !=> A==C
		for (int i = tid / translation_num;
				i < block_sz/prefetch_fraction;
				i += block_sz/translation_num)
		{
			if((init_pixel + i) >= image_size) break;

			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf( init_pixel + i   ,  projector.imgX*projector.imgY); //TODO optimize index extraction.
				xy =           ( init_pixel + i ) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + i ) % projector.imgX;
				y = floorfracf( init_pixel + i   , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

			XFLOAT real, imag;

			if(DATA3D)
				translatePixel(x, y, z, tx, ty, tz, s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);
			else
				translatePixel(x, y,    tx, ty,     s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);


			#pragma unroll
			for (int j = 0; j < eulers_per_block; j ++)
			{
				XFLOAT diff_real =  s_ref_real[eulers_per_block * i + j] - real;
				XFLOAT diff_imag =  s_ref_imag[eulers_per_block * i + j] - imag;
				diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) * s_corr[i + init_pixel % block_sz];
			}
		}
	}

	//__syncthreads();

//#define S_DIFF2S s_ref_real
	//Map
	//#pragma unroll
	//for (int j = 0; j < eulers_per_block; j ++)
	//	S_DIFF2S[tid * eulers_per_block + j] = diff2s[j];

	//__syncthreads();

	//Reduce1
	//FIXME
	//for(int j=(block_sz/2); j>0; j/=2)
	//{
	//	if(tid<j)
	//	{
	//		#pragma unroll
	//		for (int i=0; i<eulers_per_block; i++)
	//		{
	//			s_diff2s[eulers_per_block * tid + i] += s_diff2s[eulers_per_block * (tid + j) + i];
	//		}
	//	}
	//	__syncthreads();
	//}

	//Set global
	int _offset = tid % translation_num;

	if(_diff_sz >= _48KB)
	{
		#pragma unroll
		for (int i = 0; i < eulers_per_block; i ++)
			cuda_atomic_add(&g_diff2s[(blockIdx.x * eulers_per_block + i) * translation_num + _offset], diff2s[i]);
	}
	else
	{
		if(tid < translation_num)
		{
			#pragma unroll
			for (int i = 0; i < eulers_per_block; i ++)
				s_diff2s[i + eulers_per_block * tid] = diff2s[i];
		}

		__syncthreads();

		if(tid >= translation_num)
		{
			#pragma unroll
			for (int i = 0; i < eulers_per_block; i ++)
				cuda_atomic_add(&s_diff2s[i + eulers_per_block * _offset], diff2s[i]);
		}
		__syncthreads();

		if(tid < translation_num)
		{
			#pragma unroll
			for (int i = 0; i < eulers_per_block; i ++)
				g_diff2s[(blockIdx.x * eulers_per_block + i) * translation_num + tid] += s_diff2s[eulers_per_block * tid + i];
		}
	}
//#undef S_DIFF2S
}


template<bool REF3D, bool DATA3D, int block_sz, int chunk_sz>
__global__ void cuda_kernel_diff2_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		CudaProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num
		)
{
	unsigned long bid = blockIdx.x;
	unsigned long tid = threadIdx.x;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	unsigned long pixel;
	XFLOAT ref_real, ref_imag,
		shifted_real, shifted_imag,
		diff_real, diff_imag;

	__shared__ XFLOAT s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT s_outs[chunk_sz];
	// inside the padded 2D orientation gri
//	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num  = (unsigned)d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[itrans*block_sz+tid] = (XFLOAT)0.0;
		}
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num(ceilfracf(image_size,block_sz));

		for (unsigned pass = 0; pass < pass_num; pass++) // finish an entire ref image each block
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
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
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					iy = d_trans_idx[d_job_idx[bid]] + itrans;

					if(DATA3D)
						translatePixel(x, y, z, trans_x[iy], trans_y[iy], trans_z[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					else
						translatePixel(x, y, trans_x[iy], trans_y[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);

					diff_real =  ref_real - shifted_real;
					diff_imag =  ref_imag - shifted_imag;
					s[itrans*block_sz + tid] += (diff_real * diff_real + diff_imag * diff_imag) * (XFLOAT)0.5 * __ldg(&g_corr_img[pixel]);
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[itrans*block_sz+tid] += s[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
			s_outs[tid]=s[tid*block_sz]+sum_init;
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			g_diff2s[iy] = s_outs[tid];
		}
	}
}




/*
 *   	CROSS-CORRELATION-BASED KERNELS
 */

template<bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_diff2_CC_coarse(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		CudaProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned translation_num,
		int image_size,
		XFLOAT exp_local_sqrtXi2
		)
{

	int iorient = blockIdx.x;
	int itrans =  blockIdx.y;
	int tid = threadIdx.x;

    __shared__ XFLOAT s_weight[block_sz];
    s_weight[tid] = (XFLOAT)0.0;
	__shared__ XFLOAT s_norm[block_sz];
	s_norm[tid] = (XFLOAT)0.0;

	XFLOAT real, imag, ref_real, ref_imag;

	XFLOAT e0,e1,e2,e3,e4,e5,e6,e7,e8;
	e0 = __ldg(&g_eulers[iorient*9  ]);
	e1 = __ldg(&g_eulers[iorient*9+1]);
	e2 = __ldg(&g_eulers[iorient*9+2]);
	e3 = __ldg(&g_eulers[iorient*9+3]);
	e4 = __ldg(&g_eulers[iorient*9+4]);
	e5 = __ldg(&g_eulers[iorient*9+5]);
	e6 = __ldg(&g_eulers[iorient*9+6]);
	e7 = __ldg(&g_eulers[iorient*9+7]);
	e8 = __ldg(&g_eulers[iorient*9+8]);

	__syncthreads();

	unsigned pixel_pass_num( ceilfracf(image_size,block_sz) );
	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
	{
		unsigned pixel = (pass * block_sz) + tid;

		if(pixel < image_size)
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
					e0,e1,e2,e3,e4,e5,e6,e7,e8,
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					e0,e1,e3,e4,e6,e7,
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
					x,y,
					e0,e1,e3,e4,
					ref_real, ref_imag);

			if(DATA3D)
				translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);
			else
				translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);

			s_weight[tid] += (ref_real * real     + ref_imag * imag)      * __ldg(&g_corr_img[pixel]);
			s_norm[tid]   += (ref_real * ref_real + ref_imag * ref_imag ) * __ldg(&g_corr_img[pixel]);
		}
		__syncthreads();
	}


	for(int j=(block_sz/2); j>0; j/=2)
	{
		if(tid<j)
		{
			s_weight[tid] += s_weight[tid+j];
			s_norm[tid]   += s_norm[tid+j];
		}
		__syncthreads();
	}
#ifdef CUDA_DOUBLE_PRECISION
	g_diff2s[iorient * translation_num + itrans] = - ( s_weight[0] / sqrt(s_norm[0]));
#else
	g_diff2s[iorient * translation_num + itrans] = - ( s_weight[0] / sqrtf(s_norm[0]));
#endif
}

template<bool REF3D, bool DATA3D, int block_sz,int chunk_sz>
__global__ void cuda_kernel_diff2_CC_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		CudaProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		XFLOAT exp_local_sqrtXi2,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num
		)
{
	int bid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.x;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	int pixel;
	XFLOAT ref_real, ref_imag, shifted_real, shifted_imag;

	__shared__ XFLOAT      s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT   s_cc[block_sz*chunk_sz];
	__shared__ XFLOAT s_outs[chunk_sz];

	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num   = d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[   itrans*block_sz+tid] = 0.0f;
			s_cc[itrans*block_sz+tid] = 0.0f;
		}
		__syncthreads();
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num( ceilfracf(image_size,block_sz) );

		for (unsigned pass = 0; pass < pass_num; pass++) // finish an entire ref image each block
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
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
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					iy = d_trans_idx[d_job_idx[bid]] + itrans;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[iy], g_trans_y[iy], g_trans_z[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					else
						translatePixel(x, y,    g_trans_x[iy], g_trans_y[iy],                g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);

					s[   itrans*block_sz + tid] += (ref_real * shifted_real + ref_imag * shifted_imag) * __ldg(&g_corr_img[pixel]);
					s_cc[itrans*block_sz + tid] += (ref_real*ref_real + ref_imag*ref_imag) * __ldg(&g_corr_img[pixel]);
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[   itrans*block_sz+tid] += s[   itrans*block_sz+tid+j];
					s_cc[itrans*block_sz+tid] += s_cc[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
#ifdef CUDA_DOUBLE_PRECISION
			s_outs[tid]= - s[tid*block_sz] / (sqrt(s_cc[tid*block_sz]));
#else
			s_outs[tid]= - s[tid*block_sz] / (sqrtf(s_cc[tid*block_sz]));
#endif
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			g_diff2s[iy] = s_outs[tid];
		}
	}
}
#endif /* CUDA_DIFF2_KERNELS_CUH_ */
