import caffe
import lmdb
import random
import numpy as np
import cv2
class gan:
    def __init__(self,proto=''):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.solver=caffe.RMSPropSolver(proto)
        self.batch_size=100
    def get_img(self,dst):
        img=(dst.transpose((1,2,0,3))+1.0)*127.5
        img=img.reshape(3,64,10,64*10)
        img=img.transpose(0,3,2,1).reshape(3,64*10,64*10).transpose(0,2,1)
        img=img.transpose(1,2,0)
        img[img>255]=255
        img[img<0]=0
        img=img.astype(np.uint8)
        return img
    def save_img(self,idx):
        st=random.randint(0,900)
        imgs=self.get_img(self.fdata[st:st+100])
        cv2.imwrite('./image/g_%d.jpg'%(idx),imgs)
    def init_data(self):
        sam_num=2000
        dim=100
        nclass=1
        data=np.random.uniform(-1, 1, size=(sam_num , dim))
        data=data.reshape(sam_num,1,1,dim)
        label=np.zeros((sam_num,1,1,1))+1
        self.gdata=np.array(data,dtype=np.float32)
        self.glabel=np.array(label,dtype=np.float32)
        self.solver.net.set_input_arrays_with_name('data',self.gdata,self.glabel)
    def load_imgs(self):
        tdata=[]
        tlabel=[]
        env = lmdb.open('F:/face/img_align_celeba/face_small_lmdb',readonly=True)
        i=0
        with env.begin() as txn:
            cursor = txn.cursor()
            for key,value in cursor:
                i+=1
                if i>12000: break
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(value)
                flat_x=np.fromstring(datum.data,dtype=np.uint8)
                x=flat_x.reshape(datum.channels,datum.height,datum.width)/127.5
                x=x-1
                tlabel.append(datum.label)
                tdata.append(x)
        self.tdata=np.array(tdata)
        self.tlabel=np.array(tlabel)
    def generate_image(self):
        batches=10
        size=batches*self.batch_size
        self.fdata=np.zeros((size,3,64,64),dtype=np.float32)
        for i in range(batches):
            start_idx=i*self.batch_size
            self.solver.net.forward(None,None,'gdata')
            self.fdata[start_idx:start_idx+self.batch_size]=np.copy(self.solver.net.blobs['gdata'].data)
    def dis_generate_training_data(self):
        self.generate_image()
        idx=random.sample(range(len(self.tdata)),len(self.fdata))
        self.dlabel=np.zeros((len(self.fdata)*2,1,1,1),dtype=np.float32)
        l=len(self.fdata)
        self.ddata=np.zeros((l*2,3,64,64),dtype=np.float32)
        self.ddata[0:l]=self.tdata[idx]
        self.ddata[l:]=self.fdata
        self.dlabel[0:l]+=1
        self.dlabel[l:]-=1
        self.solver.net.set_input_arrays_with_name('data2',self.ddata,self.dlabel)
    def set_train_generator(self):
        self.solver.net.select_switch_with_name('sw_data',0)
        self.solver.net.select_switch_with_name('sw_label',0)
        self.solver.net.lock_net_with_idx(-1)
        self.solver.net.select_net_with_idx(3)
        self.solver.net.lock_net_with_idx(2)
        self.init_data()
        self.solver.net.forward()
    def set_train_discriminator(self):
        self.dis_generate_training_data()
        self.solver.net.select_switch_with_name('sw_data',1)
        self.solver.net.select_switch_with_name('sw_label',1)
        self.solver.net.lock_net_with_idx(-1)
        self.solver.net.select_net_with_idx(6)
        self.solver.net.forward()
    def out_loss(self,phase):
        if phase == 'gen':
            print "gloss: %f "%(self.solver.net.blobs['loss'].data)
        else:
            print "dloss: %f "%(self.solver.net.blobs['loss'].data)
    def train(self):
        steps=500000
        gstep=2
        dstep=10
        self.solver.net.clip_net_with_idx(2,0.006)
        self.set_train_generator()
        self.load_imgs()
        #self.solver.restore('./model/gan__iter_890000.solverstate')
        for sp in range(steps):
            print "gan iter: %d"%sp
            self.solver.restart()
            self.set_train_discriminator()
            self.out_loss('dis')
            self.solver.net.blobs['loss'].diff[...]=1
            for dsp in range(dstep):
                self.solver.step(10)
                self.out_loss('dis')
            self.solver.restart()
            self.set_train_generator()
            self.out_loss('gen')
            self.solver.net.blobs['loss'].diff[...]=1
            for gsp in range(gstep):
                self.solver.step(10)
                self.out_loss('gen')
            if sp % 20 == 0: self.save_img(sp)
if __name__ == '__main__':
    gn=gan('./gan_rmsprop_solver.prototxt')
    gn.train()