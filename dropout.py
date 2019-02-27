
#dropout
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA',one_hot = True)

#层级模型
def add_layer(input,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer_name'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases =tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input,Weights)+biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

# 计算百分比
def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:0.6})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:0.6})
    return result

#模型输入以及输出
keep_prob=tf.placeholder(tf.float32)
with tf.name_scope('input'):
    xs=tf.placeholder(tf.float32,[None,784],name='x_input')
    ys=tf.placeholder(tf.float32,[None,10],name='y_input')

#隐藏层
l1 = add_layer(xs,784,500,n_layer=1,activation_function=tf.nn.tanh)
prediction = add_layer(l1,500,10,n_layer=2,activation_function=tf.nn.softmax)

#损失函数(交叉熵)
with tf.name_scope('loss'):
    cross_entropy =tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',cross_entropy)

#训练方案设计
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

#初始化变量
init = tf.initialize_all_variables()

#程序开始运行
sess=tf.Session()
merged=tf.summary.merge_all()
writer = tf.summary.FileWriter("los/",sess.graph)
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.6})
    if i%50==0:
       print(computer_accuracy(mnist.test.images,mnist.test.labels))