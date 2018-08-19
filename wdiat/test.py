import caffe
import lmdb
import numpy  as np
import cv2
import os
def get_img(dst):
        img=dst*127.5+127.5
        img=img.transpose(1,2,0)#+np.array([93.594,104.7624,129.1863])
        img[img>255]=255
        img[img<0]=0
        img=img.astype(np.uint8)
        return img
def load_data():
        rdata=[]
        #env = lmdb.open('f:/d_disk/dev/face_open_lmdb',readonly=True)
        #env = lmdb.open('f:/d_disk/dev/face_male_lmdb',readonly=True)
        env = lmdb.open('F:/face/img_align_celeba/super/face_super_small_lmdb_f',readonly=True)
        #env = lmdb.open('F:/face/img_align_celeba/super/face_super_truth_lmdb_f',readonly=True)
        i = 0
        with env.begin() as txn:
            cursor = txn.cursor()
            for key,value in cursor:
                if i == 600: break
                i+=1
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(value)
                flat_x=np.fromstring(datum.data,dtype=np.uint8)
                x=flat_x.reshape(datum.channels,datum.height,datum.width)/127.5
                x=x-1
                rdata.append(x)
        num=len(rdata)-len(rdata)%100
        rdata=rdata[0:num]
        label=np.zeros((num,1,1,1))+1
        raw_data=np.array(rdata,dtype=np.float32)
        return raw_data
def load_data_istock():
        rdata=[]
        for pt in os.listdir('G:/istock/img2/align/'):
            img=cv2.imread('G:/istock/img2/align/%s'%pt)
            x=img.transpose(2,0,1)/127.5
            x=x-1
            rdata.append(x)
        num=len(rdata)-len(rdata)%100
        rdata=rdata[0:num]
        label=np.zeros((num,1,1,1))+1
        raw_data=np.array(rdata,dtype=np.float32)
        return raw_data
def load_data_other():
        rdata=[]
        for pt in os.listdir('./source/'):
            img=cv2.imread('./source/%s'%pt)
            x=img.transpose(2,0,1)/127.5
            x=x-1
            rdata.append(x)
        num=len(rdata)-len(rdata)%20
        rdata=rdata[0:num]
        label=np.zeros((num,1,1,1))+1
        raw_data=np.array(rdata,dtype=np.float32)
        return raw_data
if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net=caffe.Net('./super_deploy.prototxt',caffe.TEST)
    #net=caffe.Net('./test_deploy.prototxt',caffe.TEST)
    data=load_data()
    #net.copy_from('./model/wdiat/mouth_close.caffemodel')
    #net.copy_from('./model/wdiat/male2female.caffemodel')
    net.copy_from('./model/wdiat/super_man.caffemodel')
    idx=0
    
    #data=load_data_istock()
    bn = data.shape[0]/20
    label=np.zeros((data.shape[0],1,1,1),dtype=np.float32)
    net.set_input_arrays_with_name('data',data,label)
    for i in range(bn):
        net.forward()
        for j in range(20):
            cv2.imwrite('./imgs2/%d_s.jpg'%(i*20+j),get_img(net.blobs['data'].data[j]))
            #cv2.imwrite('./super_res/%d_gg.jpg'%(i*20+j),get_img(net.blobs['a_gdata'].data[j]))
            cv2.imwrite('./imgs2/%d_g%d.jpg'%(i*20+j,idx),get_img(net.blobs['gdata_a'].data[j]))