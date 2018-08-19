#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "mask_layer.hpp"
//#include "thrust/host_vector.h"
namespace caffe {


	template <typename Dtype>
	__global__ void mask_forward_kernel(const int nthreads, const Dtype* const bottom_data,
		Dtype* const top_data, const int base, const int channel) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int num = index / base;
			int hw = index % base;
			
			if (bottom_data[index] > 0.5){
				for (int i = 0; i < channel; i++)
				{
					top_data[(num * channel+i) * base + hw] = 1;
				}
			}
			else{
				for (int i = 0; i < channel; i++)
				{
					top_data[(num * channel + i) * base + hw] = 0;
				}
			}
		}
	}
	template <typename Dtype>
	void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype * bottom_data = bottom[0]->gpu_data();
		
		mask_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS >> >
			(count_, bottom_data, top[0]->mutable_gpu_data(),height_*width_,channel_out_);
		CUDA_POST_KERNEL_CHECK;
		Dtype sum_ratio;
		caffe_gpu_asum(top[0]->count(), top[0]->gpu_data(), &sum_ratio);
		this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / (num_*width_*height_*channel_out_);
	}
	template <typename Dtype>
	__global__ void mask_backward_kernel(const int nthreads, Dtype* const bottom_diff,
		const Dtype* const top_diff, const int base,const int channel) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int num = index / base;
			int hw = index % base;
			//bottom_diff[index] = top_diff[num*3*base+hw];
			//bottom_diff[index] += top_diff[num * 3 * base + base+hw];
			//bottom_diff[index] += top_diff[num * 3 * base + 2*base + hw];
			for (int i = 0; i < channel; i++)
			{
				bottom_diff[index]+=top_diff[(num * channel + i) * base + hw];
			}
		}
	}
	template <typename Dtype>
	void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
		mask_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS >> >
				(count_, bottom_diff, top[0]->gpu_diff(),height_*width_,channel_out_);
		CUDA_POST_KERNEL_CHECK;
		Dtype sum_ratio = this->blobs_[0]->cpu_data()[0];
		if (sum_ratio>ratio_)
		{
			caffe_gpu_add_scalar(bottom[0]->count(), weight_, bottom[0]->mutable_gpu_diff());
		}
		else{
			caffe_gpu_add_scalar(bottom[0]->count(), -weight_, bottom[0]->mutable_gpu_diff());
		}

	}

	INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);
}  // namespace caffe
