#CNN,数据保存和复原
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA',one_hot = True)

def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:0.6})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:0.6})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):            #卷积层运行方式
    #strides 步长(1（必定），1（x步长），1（y步长），1（必定）)
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):        #池化层运行方式
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

keep_prob=tf.placeholder(tf.float32)    #dropout的
xs=tf.placeholder(tf.float32,[None,784],name='x_input') #输入数据
ys=tf.placeholder(tf.float32,[None,10],name='y_input')  #输出数据
#784=28*28(分辨率)，1个通道,-1数量（全部）   
x_image=tf.reshape(xs,[-1,28,28,1])     #转化为图片

#定义卷积层1(5*5的大小，insize 1,outsize 32) 32个卷积核
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                      #output 14*14*32

#定义卷积层2(5*5的大小，insize 32,outsize 128)
W_conv2 = weight_variable([5,5,32,128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                      #output 7*7*64


# func1全连接层（7*7*128个输入数据，512个神经元）
W_fc1 = weight_variable([7*7*128,512])
b_fc1 = bias_variable([512])
#将卷积结果平面化
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*128])
#进入全连接层
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fcl_drop = tf.nn.dropout(h_fcl,keep_prob)

#func2全连接层（512个输入数据，64个神经元）
W_fc2 = weight_variable([512,64])
b_fc2 = bias_variable([64])
h_fc2 = tf.nn.relu(tf.matmul(h_fcl_drop,W_fc2)+b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

#func3全连接层（64个输入数据，10个神经元）
W_fc3 = weight_variable([64,10])
b_fc3 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc2_drop,W_fc3)+b_fc3)

#loss函数，交叉熵
cross_entropy =tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#训练方法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#保存数据
saver = tf.train.Saver()
sess=tf.Session()
saver.restore(sess,"my_data/1.ckpt")
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.6})
    if i%50==0:
       print(computer_accuracy(mnist.test.images,mnist.test.labels))

saver.save(sess,"my_data/1.ckpt")
print("saved")