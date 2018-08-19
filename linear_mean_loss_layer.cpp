#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/linear_mean_loss_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void LinearMeanLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		n_ = bottom[0]->count();
		diff_.Reshape(1, 1, 1, 1);
		direction_ = 1.0;
	}
	template <typename Dtype>
	void LinearMeanLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);  
		top[0]->Reshape(loss_shape);
		n_ = bottom[0]->count();
		ones_.Reshape(n_, 1, 1, 1);
		caffe_set(n_, Dtype(1.0), ones_.mutable_cpu_data());
	}
	template <typename Dtype>
	void LinearMeanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		MeanLossParameter mp = this->layer_param_.mean_loss_param();
		switch (mp.method()){
		case MeanLossParameter_LossMethod_LONE:
		{
			Dtype loss = caffe_cpu_dot(n_, ones_.cpu_data(), bottom[0]->cpu_data());
			top[0]->mutable_cpu_data()[0] = loss / n_;
			direction_ = -bottom[1]->cpu_data()[0];
			break;
		}
		case MeanLossParameter_LossMethod_LTWO:
		{
			Dtype loss = caffe_cpu_dot(n_, bottom[0]->cpu_data(), bottom[0]->cpu_data());
			top[0]->mutable_cpu_data()[0] = loss /  bottom[0]->num();
			break;
		}
		default:
			break;
		}
		
	}

	template <typename Dtype>
	void LinearMeanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		MeanLossParameter mp = this->layer_param_.mean_loss_param();

		switch (mp.method()){
		case MeanLossParameter_LossMethod_LONE:
		{
			Dtype alpha = top[0]->cpu_diff()[0] * direction_ / n_;
			caffe_set(n_, alpha, bottom[0]->mutable_cpu_diff());
			break;
		}
		case MeanLossParameter_LossMethod_LTWO:
		{
			Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
			caffe_cpu_axpby(n_, alpha, bottom[0]->cpu_data(), Dtype(0), bottom[0]->mutable_cpu_diff());
			break;
		}
		default:
			break;
		}


	}

#ifdef CPU_ONLY
	STUB_GPU(LinearMeanLossLayer);
#endif

	INSTANTIATE_CLASS(LinearMeanLossLayer);
	REGISTER_LAYER_CLASS(LinearMeanLoss);

}  // namespace caffe
