#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "mask_layer.hpp"
namespace caffe {
	template <typename Dtype>
	void MaskLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		ImpMapParameter rm = this->layer_param_.imp_map_param();
		//LOG(INFO) << "1L";
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
		weight_ = rm.weight();
		ratio_ = rm.cmp_ratio();
		channel_out_=rm.channel_out();
	}
	template <typename Dtype>
	void MaskLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//LOG(INFO) << "4L";
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		channel_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		count_ = bottom[0]->count();
		//LOG(INFO) << "5L";
		one_multiper_.Reshape(1, 1, 1, channel_out_);
		//LOG(INFO) << "3L";
		caffe_set(channel_out_, Dtype(1.0), one_multiper_.mutable_cpu_data());
		top[0]->Reshape(num_, channel_out_, height_, width_);
		CHECK_EQ(channel_, 1);
		//LOG(INFO) << height_ << ": " << width_ << ": " << channel_ << ": " << num_ << ": " << count_;
	}
	template <typename Dtype>
	void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype *  top_data = top[0]->mutable_cpu_data();
		const Dtype *  bottom_data = bottom[0]->cpu_data();
		int base = width_*height_;
		//LOG(INFO) << num_ << ": " << base;
		for (int i = 0; i < num_; i++)
		{
			for (int k = 0; k < channel_out_; k++)
			{

				for (int j = 0; j<base; j++)
				{
					if (bottom_data[j]>Dtype(0.5))
						top_data[j] = 1;
					else
						top_data[j] = 0;
					//LOG(INFO) << j << ": " << bottom_data[j];

				}
				top_data += base;
			}
			bottom_data += base;

		}
		Dtype sum_ratio = top[0]->asum_data();
		this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / (num_*width_*height_*channel_out_);
	}
	template <typename Dtype>
	void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype * one_mult = one_multiper_.cpu_data();
		const Dtype * top_diff = top[0]->cpu_diff();
		Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();

		for (int i = 0; i < num_; i++)
		{
		caffe_cpu_gemv<Dtype>(CblasTrans, channel_out_, width_*height_, 1.,
		top_diff, one_mult, 0.,bottom_diff);
		top_diff = top_diff + channel_out_*width_*height_;
		bottom_diff = bottom_diff + width_*height_;
		}
		
		Dtype sum_ratio = this->blobs_[0]->cpu_data()[0];
		//this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / (num_*width_*height_*channel_out_);
		if (sum_ratio>ratio_)
		{
			caffe_add_scalar(bottom[0]->count(), weight_, bottom[0]->mutable_cpu_diff());
		}
		else{
			caffe_add_scalar(bottom[0]->count(), -weight_, bottom[0]->mutable_cpu_diff());
		}

	}

#ifdef CPU_ONLY
	STUB_GPU(MaskLayer);
#endif

	INSTANTIATE_CLASS(MaskLayer);
	REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
