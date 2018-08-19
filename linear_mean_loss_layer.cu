#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/linear_mean_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void LinearMeanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		MeanLossParameter mp = this->layer_param_.mean_loss_param();
		switch (mp.method()){
		case MeanLossParameter_LossMethod_LONE:
		{
			Dtype loss;
			caffe_gpu_dot(n_, ones_.gpu_data(), bottom[0]->gpu_data(), &loss);
			top[0]->mutable_cpu_data()[0] = loss / n_;
			direction_ = -bottom[1]->cpu_data()[0];
			break;
		}
		case MeanLossParameter_LossMethod_LTWO:
		{
			Dtype loss;
			caffe_gpu_dot(n_, bottom[0]->gpu_data(), bottom[0]->gpu_data(),&loss);
			//LOG(INFO) << "1L:"<<loss;
			top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
			break;
		}
		default:
			break;
		}
		
	}

	template <typename Dtype>
	void LinearMeanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		MeanLossParameter mp = this->layer_param_.mean_loss_param();
		switch (mp.method()){
		case MeanLossParameter_LossMethod_LONE:
		{
			Dtype alpha = top[0]->cpu_diff()[0] * direction_ / n_;
			caffe_gpu_set(n_, alpha, bottom[0]->mutable_gpu_diff());
			break;
		}
		case MeanLossParameter_LossMethod_LTWO:
		{
			Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
			caffe_gpu_axpby(n_, alpha, bottom[0]->gpu_data(), Dtype(0), bottom[0]->mutable_gpu_diff());
			break;
		}
		default:
			break;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(LinearMeanLossLayer);

}  // namespace caffe
