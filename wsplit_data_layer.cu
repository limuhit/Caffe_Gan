#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "wsplit_data_layer.hpp"
namespace caffe {

	template <typename Dtype>
	void WSplitDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//top[0]->ShareData(*bottom[0]);
		//top[1]->ShareData(*bottom[0]);
	}

	template <typename Dtype>
	void WSplitDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype a,b,alpha;
		int cnt = bottom[0]->count();
		caffe_gpu_asum(cnt, top[0]->gpu_diff(), &a);
		caffe_gpu_asum(cnt, top[1]->gpu_diff(), &b);
		if (b < 0.000001)b = Dtype(1.0);
		alpha = a / b*weight_;
		caffe_gpu_memcpy(cnt * sizeof(Dtype), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		caffe_gpu_axpby(cnt, alpha, top[1]->gpu_diff(), Dtype(1.0), bottom[0]->mutable_gpu_diff());
	}

	INSTANTIATE_LAYER_GPU_FUNCS(WSplitDataLayer);

}  // namespace caffe
