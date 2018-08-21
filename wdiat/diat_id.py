import caffe
import lmdb
import random
import numpy as np
import cv2
import time
class semi_gan:
    def __init__(self,proto=''):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.solver=caffe.RMSPropSolver(proto)
        self.batch_size=30
        self.label_a=np.zeros((self.batch_size*2,1,1,1),dtype=np.float32)
        self.label_b=np.zeros((self.batch_size*2,1,1,1),dtype=np.float32)
    def get_img(self,dst):
        img=dst.transpose((1,2,0,3))*127.5+127.5
        img=img.reshape(3,128,128*10)
        img=img.transpose(1,2,0)#+np.array([93.594,104.7624,129.1863])
        img[img>255]=255
        img[img<0]=0
        img=img.astype(np.uint8)
        return img
    def save_img(self,idx,bt=0):
        imgs=np.zeros((128*2,128*10,3),dtype=np.uint8)
        imgs[0:128,:]=self.get_img(self.solver.net.blobs['data'].data[bt*10:bt*10+10])
        imgs[128:,:]=self.get_img(self.solver.net.blobs['gdata_a'].data[bt*10:bt*10+10])
        cv2.imwrite('./img/g_%d_%d.jpg'%(idx,bt),imgs)
    def save_mask(self,idx):
        imgs=np.zeros((128*2,128*10,3),dtype=np.uint8)
        imgs[0:128,:]=self.get_img(self.solver.net.blobs['mdata_gf'].data[:10])
        imgs[128:,:]=self.get_img(self.solver.net.blobs['mdata_gf_inv'].data[:10])
        cv2.imwrite('./img/m_%d.jpg'%(idx),imgs)
    def load_data(self):
        rdata=[]
        env = lmdb.open('f:/d_disk/dev/face_male_lmdb',readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key,value in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(value)
                flat_x=np.fromstring(datum.data,dtype=np.uint8)
                x=flat_x.reshape(datum.channels,datum.height,datum.width)/127.5
                x=x-1
                rdata.append(x)
        num=len(rdata)-len(rdata)%100
        rdata=rdata[0:num]
        label=np.zeros((num,1,1,1))+1
        self.raw_data=np.array(rdata,dtype=np.float32)
    def init_data(self):
        sam_num=self.batch_size*2
        idx=random.sample(range(len(self.raw_data)),sam_num)
        label=np.zeros((sam_num,1,1,1))+1
        self.gdata=self.raw_data[idx]
        self.glabel=np.array(label,dtype=np.float32)
        self.solver.net.set_input_arrays_with_name('data',self.gdata,self.glabel)
    def load_imgs(self):
        tdata=[]
        env = lmdb.open('f:/d_disk/dev/face_female_lmdb',readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key,value in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(value)
                flat_x=np.fromstring(datum.data,dtype=np.uint8)
                x=flat_x.reshape(datum.channels,datum.height,datum.width)/127.5
                x=x-1
                tdata.append(x)
        self.tdata=np.array(tdata)
    def generate_image(self):
        batches=2
        size=batches*self.batch_size
        self.fdata=np.zeros((size,3,128,128),dtype=np.float32)
        for i in range(batches):
            start_idx=i*self.batch_size
            self.solver.net.forward(None,None,'gdata_a')
            self.fdata[start_idx:start_idx+self.batch_size]=np.copy(self.solver.net.blobs['gdata_a'].data)
    def generate_image_semi(self):
        batches=2
        size=batches*self.batch_size
        self.fdata=np.zeros((size,3,128,128),dtype=np.float32)
        self.sdata=np.zeros((size,3,128,128),dtype=np.float32)
        idx=random.sample(range(len(self.tdata)),size)
        self.sdata[:]=self.tdata[idx]
        for i in range(batches):
            start_idx=i*self.batch_size
            self.solver.net.blobs['gdata_a_0'].data[...]=self.sdata[start_idx:start_idx+self.batch_size]
            self.solver.net.forward(None,'b_conv1','b_gdata')
            self.fdata[start_idx:start_idx+self.batch_size]=np.copy(self.solver.net.blobs['b_gdata'].data)
        self.solver.net.set_input_arrays_with_name('data',self.fdata,self.label_a)
        self.solver.net.set_input_arrays_with_name('data2',self.sdata,self.label_b)
    def dis_generate_training_data(self):
        self.generate_image()
        idx=random.sample(range(len(self.tdata)),len(self.fdata))
        self.dlabel=np.zeros((len(self.fdata)*2,1,1,1),dtype=np.float32)
        l=len(self.fdata)
        self.ddata=np.zeros((l*2,3,128,128),dtype=np.float32)
        self.ddata[0:l]=self.tdata[idx]
        self.ddata[l:]=self.fdata
        self.dlabel[0:l]+=1
        self.dlabel[l:]-=1
        self.solver.net.set_input_arrays_with_name('data3',self.ddata,self.dlabel)
    def clear_diffs(self):
        self.solver.net.blobs['gdata_a_5'].diff[...]=0
        self.solver.net.blobs['gdata_a_4'].diff[...]=0
        self.solver.net.blobs['gdata_a_3'].diff[...]=0
        self.solver.net.blobs['gdata_a_2'].diff[...]=0
        self.solver.net.blobs['gdata_a_1'].diff[...]=0
        self.solver.net.blobs['gdata_a_0'].diff[...]=0
    def extract_vgg_id_feature(self):
        num_batch=2
        self.id_feature = np.zeros((num_batch*self.batch_size,1,1,4096),dtype=np.float32)
        for i in range(num_batch):
            self.solver.net.blobs['gdata_a_5'].data[...]=self.gdata[self.batch_size*i:self.batch_size*(i+1)]
            self.solver.net.forward(None,'gdata_resize','face_fc7')
            self.id_feature[self.batch_size*i:self.batch_size*(i+1)]=np.copy(self.solver.net.blobs['face_fc7'].data.reshape(self.batch_size,1,1,4096))
        self.solver.net.set_input_arrays_with_name('data4',self.id_feature,self.glabel)
        self.solver.net.forward()
    def set_train_generator(self):
        self.solver.net.select_switch_with_name('sw_data',0)
        self.solver.net.select_switch_with_name('sw_label',0)
        self.solver.net.lock_net_with_idx(-1)
        self.solver.net.select_net_with_idx(51+128)#115
        self.solver.net.lock_net_with_idx(16)
        self.solver.net.lock_net_with_idx(32)
        self.solver.net.lock_net_with_idx(128)
        self.clear_diffs()
        self.init_data()
        self.extract_vgg_id_feature()
    def set_train_discriminator(self):
        self.dis_generate_training_data()
        self.solver.net.select_switch_with_name('sw_data',1)
        self.solver.net.select_switch_with_name('sw_label',1)
        self.solver.net.lock_net_with_idx(-1)
        self.solver.net.select_net_with_idx(24)
    def set_train_tv(self):
        self.solver.net.lock_net_with_idx(-1)
        self.solver.net.select_net_with_idx(65)
        self.solver.net.lock_net_with_idx(64)
        self.clear_diffs()
    def set_train_super(self):
        self.solver.net.lock_net_with_idx(-1)
        self.solver.net.select_net_with_idx(2)
        self.generate_image_semi()
        self.solver.net.lock_net_with_idx(-1)
        self.solver.net.select_net_with_idx(5)
        self.clear_diffs()
    def out_loss(self,phase):
        if phase == 'gen':
            print "gloss: %f id_per_loss: %f id_loss: %f semi_loss: %f mask: %f pix_loss: %f" %(self.solver.net.blobs['loss_gan'].data,
                        self.solver.net.blobs['loss_per'].data,
                        self.solver.net.blobs['loss_id'].data,
                        self.solver.net.blobs['loss_ab'].data,
                        self.solver.net.params['mdata_l'][0].data,
                        self.solver.net.blobs['pix_loss'].data)
        else:
            print "dloss: %f "%(self.solver.net.blobs['loss_gan'].data)
    def get_gaussian_kernel(self,sz,sig):
        res=np.zeros((2*sz-1,2*sz-1))
        res[sz-1,sz-1]=1
        kernel=cv2.GaussianBlur(res,(sz,sz),sig,0)
        return kernel[sz/2:sz/2+sz,sz/2:sz/2+sz]
    def train(self):
        steps=500000
        gstep=3
        dstep=12
        sstep=1
        tstep=3
        self.load_data()
        self.solver.net.clip_net_with_idx(16,0.031)
        
        self.load_imgs()
        self.solver.net.copy_from('./denoise3.caffemodel')
        self.solver.net.copy_from('./pre_iter_2000.caffemodel')
        self.solver.net.copy_from('./vgg_face_trans.caffemodel')
        #self.solver.net.copy_from('./per/per_200.caffemodel')
        #self.solver.net.copy_from('./vgg_params.caffemodel')
        #self.solver.net.copy_from('./model/semi_iter_40000.caffemodel')
        #self.solver.net.copy_from('./wmodel/gan_iter_60000.caffemodel')
        #self.solver.net.select_net_with_idx(16)
        #self.solver.net.lock_net_with_idx(1)
        #self.solver.net.lock_net_with_idx(2)
        #self.solver.reinit()
        kn2=self.get_gaussian_kernel(5,2.0)
        self.solver.net.params['m_conv5_gf'][0].data[0,0]=kn2
        kn2=self.get_gaussian_kernel(9,2.0)
        self.solver.net.params['mdata_gf'][0].data[0,0]=kn2
        self.solver.net.params['mdata_gf'][0].data[1,1]=kn2
        self.solver.net.params['mdata_gf'][0].data[2,2]=kn2
        gsum=lambda xa:np.sum(np.abs(xa))
        la=0
        lb=0
        for sp in xrange(0,steps):
            self.set_train_generator()
            print "gan iter: %d"%sp
            self.solver.net.blobs['loss_gan'].diff[...]=1
            self.set_train_discriminator()
            self.out_loss('dis')
            for dsp in range(dstep):
                self.solver.step(2)
                self.out_loss('dis')
            self.set_train_generator()
            self.out_loss('gen')
            self.solver.step(2)
            df=8000./gsum(self.solver.net.blobs['gdata_a'].diff)
            if lb-la>1: 
                self.solver.net.blobs['loss_gan'].diff[...]=1
            else:  
                self.solver.net.blobs['loss_gan'].diff[...]=df
            print self.solver.net.blobs['loss_gan'].diff[...]
            for gsp in range(gstep):
                self.solver.step(2)
                self.out_loss('gen')
                print gsum(self.solver.net.blobs['ds_conv2_2'].diff)
                print gsum(self.solver.net.blobs['ds_conv2_2_s0'].diff)
                print gsum(self.solver.net.blobs['ds_conv2_2_s1'].diff)
                if gsp == 0:
                    la = np.copy(self.solver.net.blobs['loss_gan'].data)
                elif gsp == gstep-1:
                    lb = np.copy(self.solver.net.blobs['loss_gan'].data)
            print la,lb
            print "tv"
            self.set_train_tv()
            self.solver.net.forward()
            val=self.solver.net.blobs['tv_loss'].data*0.3-6
            if val<0.1:val=0.1
            print val
            self.solver.net.blobs['tv_loss'].diff[...]=val
            self.solver.net.set_loss_by_name('tv_loss',val)
            for ssp in range(tstep):
                self.solver.step(1)
                if self.solver.net.blobs['tv_loss'].data<20:break
                print self.solver.net.blobs['tv_loss'].data
            if sp % 2 == 0: self.save_img(sp)
            if sp % 2 == 0: self.save_mask(sp)
            print "semi"
            self.set_train_super()
            for ssp in range(sstep):
                self.solver.step(1)
                print self.solver.net.blobs['loss_ba'].data
            if sp%100==0 and sp>0 : self.solver.net.save('./per/per_%d.caffemodel'%sp)
if __name__ == '__main__':
    gn=semi_gan('./semi_gan_rmsprop_solver.prototxt')
    gn.train()