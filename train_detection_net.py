'''
Implementation of "A Convolutional Neural Network Cascade for Face Detection "
Paper : https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf
Author : Dennis Liu
Modify : 2017/11/10

Description :   The example of training detection nets .

'''
import cv2
import numpy as np
import tensorflow as tf
    
import model


from data import DataSet
from dataset.fddb_crawler import parse_data_info


def train_det_net():
    # get all training sample
    data_info = parse_data_info(only_positive = False)
    # data_info = [<image-path str>,[<nonface/face int>,<pattern-id int>]]


    # training configuration
    batch = 500
    size = (48,48,3)
    start_epoch = 0
    end_epoch = 3
    train_validation_rate = 0.9 # training set / all sample

    # load the pretrained model , set None if you don't have
    pretrained =   'models/det_nets_3.ckpt'

    # load data iterater
    dataset = DataSet(data_info,train_rate = train_validation_rate)
    _ , train_op , val_op , next_ele = dataset.get_iterator(batch,size)

    
    # load network
    # learning rate is great impact in training models
    net_12 = model.detect_12Net(lr = 0.001,size = (12,12,3))  
    net_24 = model.detect_24Net(lr = 0.001,size = (24,24,3))
    net_48 = model.detect_48Net(lr = 0.001,size = (48,48,3))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    if pretrained:
        saver.restore(sess , pretrained)
    else:
        sess.run(tf.global_variables_initializer())    
    
    

    for epoch in xrange(start_epoch,end_epoch):
        loss = 0
        iteration = 0
        sess.run(train_op)
        # get each element of the training dataset until the end is reached
        while True:
            try:
                # default of the size returned from data iterator is 48
                inputs,clss ,pattern = sess.run(next_ele)
                # <ndarray> , <0/1> , <one-hot of 45-class>


                clss = clss.reshape(batch,2)
                pattern = pattern.reshape(batch,45)
                

                # resize image to fit each net
                inputs_12 = np.array([cv2.resize(img,(net_12.size[0],net_12.size[1])) for img in inputs])
                inputs_24 = np.array([cv2.resize(img,(net_24.size[0],net_24.size[1])) for img in inputs])
                inputs_48 = np.array([cv2.resize(img,(net_48.size[0],net_48.size[1])) for img in inputs])

                # forward 12net
                net_12_fc = net_12.get_fc(inputs_12)
                
                # forward 24net
                net_24_fc = net_24.get_fc(inputs_24,net_12_fc)
        
                train_nets = [net_12,net_24,net_48]
                net_feed_dict = {net_12.inputs:inputs_12 , net_12.targets:clss,\
                                net_24.inputs:inputs_24 , net_24.targets:clss,net_24.from_12:net_12_fc,\
                                net_48.inputs:inputs_48 , net_48.targets:clss,net_48.from_24:net_24_fc}

                # training net
                sess.run([net.train_step for net in train_nets],\
                        feed_dict = net_feed_dict)
                # loss computation
                losses = sess.run([net.loss for net in train_nets],\
                        feed_dict = net_feed_dict)

                if iteration % 100 == 0:
                    net_12_eva = net_12.evaluate(inputs_12,clss)
                    net_12_acc = sum(net_12_eva)/len(net_12_eva)
                    net_24_eva = net_24.evaluate(inputs_24,clss,net_12_fc)
                    net_24_acc = sum(net_24_eva)/len(net_24_eva)
                    net_48_eva = net_48.evaluate(inputs_48,clss,net_24_fc)
                    net_48_acc = sum(net_48_eva)/len(net_48_eva)
                    print ('Training Epoch {} --- Iter {} --- Training Accuracy:  {}%,{}%,{}% --- Training Loss: {}'\
                            .format(epoch , iteration , net_12_acc , net_24_acc , net_48_acc  , losses))
                        

                iteration += 1
            except tf.errors.OutOfRangeError:
                # print("End of training dataset.")
                break
        
        # get each element of the validation dataset until the end is reached
        sess.run(val_op)
        net_12_acc = []
        net_24_acc = []
        net_48_acc = []
        while True:
            try:
                # the size returned from data iterator is 48
                inputs,clss ,pattern = sess.run(next_ele)
                clss = clss.reshape(batch,2)
                
                # resize image to fit each net
                inputs_12 = np.array([cv2.resize(img,(net_12.size[0],net_12.size[1])) for img in inputs])
                inputs_24 = np.array([cv2.resize(img,(net_24.size[0],net_24.size[1])) for img in inputs])
                inputs_48 = np.array([cv2.resize(img,(net_48.size[0],net_48.size[1])) for img in inputs])

                # forward 12net
                net_12_fc = net_12.get_fc(inputs_12)
                
                # forward 24net
                net_24_fc = net_24.get_fc(inputs_24,net_12_fc)

                net_12_eva = net_12.evaluate(inputs_12,clss)
                net_24_eva = net_24.evaluate(inputs_24,clss,net_12_fc)
                net_48_eva = net_48.evaluate(inputs_48,clss,net_24_fc)
                for i in range(len(net_12_eva)):
                    net_12_acc.append(net_12_eva[i])
                    net_24_acc.append(net_24_eva[i])
                    net_48_acc.append(net_48_eva[i])
            except tf.errors.OutOfRangeError:
                # print("End of validation dataset.")
                break

        print ('Validation Epoch {}  Validation Accuracy:  {}%,{}%,{}%'\
                            .format(epoch , sum(net_12_acc)/len(net_12_acc),\
                                            sum(net_24_acc)/len(net_24_acc),\
                                            sum(net_48_acc)/len(net_48_acc)))

        saver = tf.train.Saver()
        save_path = saver.save(sess, "models/48_net_{}.ckpt".format(epoch))
        print ("Model saved in file: ", save_path)

if __name__ == "__main__":
    train_det_net()