#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "wsplit_data_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void WSplitDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		WSplitParameter ws = this->layer_param_.wsplit_param();
		weight_ = ws.weight();
	}
	template <typename Dtype>
	void WSplitDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		width_ = bottom[0]->width();
		height_ = bottom[0]->height();
		channel_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		
		for (int i = 0; i < top.size(); i++)
		{
			top[i]->Reshape(num_, channel_, height_, width_);
			top[i]->ShareData(*bottom[0]);
		}
	}
	
	template <typename Dtype>
	void WSplitDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		
		
	}

	template <typename Dtype>
	void WSplitDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		
	}

#ifdef CPU_ONLY
	STUB_GPU(WSplitDataLayer);
#endif

	INSTANTIATE_CLASS(WSplitDataLayer);
	REGISTER_LAYER_CLASS(WSplitData);

}  // namespace caffe
