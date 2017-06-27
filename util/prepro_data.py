import numpy as np
from vqaTools.vqa import VQA
import random
import os
import word2vec
import sklearn
import shutil
import cv2
import gc
import time
from glob import glob
from sklearn.model_selection import train_test_split

#data provider for batch training
class Batch_generator:
    def __init__(self,img_rows,img_cols,batch_size,imgDir,mapDir,annFile,quesFile,embeddingDir,val_split=0.2):
        print 'Initializing batch generator...'
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.vqa = VQA(annFile, quesFile)
        self.batch_size = batch_size
        self.imgdir = imgDir
        print 'Loading word embedding dictionary...'
        self.embedding_model = word2vec.load(embeddingDir,encoding='ISO-8859-1') #default googleNews embedding
        self.Q, self.Img, self.target = self.init_data(mapDir)
        self.training_idx, self.val_idx = self.separate_set(len(self.Q),val_split)
        self.iter_num = len(self.training_idx)/self.batch_size
        self.mean = np.load('mean.npy')
        self.std = np.load('std.npy')

    #storing question data and image directories
    def init_data(self,imgdir):
        file_ = glob(os.path.join(imgdir,'*.png'))
        question = []
        image = []
        target = []
        for cur in file_:
            if not cur[-5] == str(1):
                continue
            target.append(cur)
            cur_qid = int(os.path.basename(cur)[:-6])
            ans = self.vqa.loadQA(cur_qid)
            cur_Q = str(self.vqa.qqa[cur_qid]['question'])
            cur_Q = cur_Q.replace('?','').replace('.','').replace(',','').split()
            question.append(cur_Q)
            img_id = ans[0]['image_id']
            image.append(img_id)

        return question,image,target

    #separate data for training and validation
    def separate_set(self,data_size,test_size):
        idx = np.arange(data_size)
        training_idx, val_idx = train_test_split(idx,test_size=test_size,random_state=33)
        return training_idx,val_idx

    def get_batch(self,batch_nb,mode):
        if mode == 'train':
            cur_idx = self.training_idx[batch_nb]
        else:
            cur_idx = self.val_idx[batch_nb]
        cur_Q = self.Q[cur_idx]
        cur_Q = convert_vec(cur_Q,self.embedding_model)
        cur_Q = cur_Q.reshape((1,cur_Q.shape[0],cur_Q.shape[1]))
        cur_Q = cur_Q.astype('float32')

        cur_img = 'COCO_' + 'train2014' + '_'+ str(self.Img[cur_idx]).zfill(12) + '.jpg'
        cur_img = os.path.join(self.imgdir,cur_img)
        cur_img = cv2.imread(cur_img)
        cur_img = cv2.resize(cur_img, (self.img_cols, self.img_rows),interpolation = cv2.INTER_LINEAR)
        cur_img = cur_img.transpose((2,0,1))
        cur_img = cur_img.astype('float32')
        for i in xrange(3):
            cur_img[i,:,:] = (cur_img[i,:,:]-self.mean[i])/self.std[i]
        cur_img = cur_img.reshape((1,cur_img.shape[0],cur_img.shape[1],cur_img.shape[2]))

        cur_target = cv2.imread(self.target[cur_idx])
        cur_target = cv2.resize(cur_target, (self.img_cols/32, self.img_rows/32),interpolation = cv2.INTER_LINEAR)
        cur_target = cur_target[:,:,0]
        cur_target = cur_target.astype('float32')
        flag = 1 #use flag to control all zero saliency maps
        if np.sum(cur_target)==0:
            flag = 0
        cur_target = cur_target/np.sum(cur_target)

        cur_target = cur_target.reshape((1,cur_target.shape[0],cur_target.shape[1]))

        if mode == 'train' and batch_nb == self.iter_num-1:
            self.shuffle()

        return cur_img,cur_Q,cur_target,flag

    def shuffle(self):
        random.shuffle(self.training_idx)

#convert string to word vector
def convert_vec(sentence,embedding_model,simple=False):
    for count,cur in enumerate(sentence):
        if simple == False:
            if count == 0:
                if cur in embedding_model:
                    vec = embedding_model[cur]
                elif cur.lower() in embedding_model:
                    vec = embedding_model[cur.lower()]
                elif cur.title() in embedding_model:
                    vec = embedding_model[cur.title()]
                else:
                    vec = embedding_model['UNK']
                vec = vec.reshape((1,len(vec)))
            else:
                if cur in embedding_model:
                    tmp = embedding_model[cur]
                elif cur.lower() in embedding_model:
                    tmp = embedding_model[cur.lower()]
                elif cur.title() in embedding_model:
                    tmp = embedding_model[cur.title()]
                else:
                    tmp = embedding_model['UNK']
                tmp = tmp.reshape((1,len(tmp)))
                vec = np.append(vec,tmp,axis=0)
        else:
            if count == 0:
                if cur.lower() in embedding_model:
                    vec = embedding_model[cur.lower()]
                else:
                    vec = embedding_model['UNK']
                vec = vec.reshape((1,len(vec)))
            else:
                if cur.lower() in embedding_model:
                    tmp = embedding_model[cur.lower()]
                else:
                    tmp = embedding_model['UNK']
                tmp = tmp.reshape((1,len(tmp)))
                vec = np.append(vec,tmp,axis=0)
    return vec
