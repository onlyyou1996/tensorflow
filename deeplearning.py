#神经元网络
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

#输入输出以及噪声
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

#模型输入以及输出
with tf.name_scope('input'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

#输入层以及隐藏层
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,n_layer=2,activation_function=None)

#损失函数
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)

#训练步长设计
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
init = tf.initialize_all_variables()

#程序开始运行
sess=tf.Session()
merged=tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)

#画图
fig = plt.figure()              #建图
ax=fig.add_subplot(1,1,1)       
ax.scatter(x_data,y_data)       #录入数据
plt.ion()                       #连续画图
plt.show()                      #开始画图

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)                                        #tensorboard输出
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))           #文字输出
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        try:
            ax.lines.remove(lines[0])                       #删除线
        except Exception:
            pass
        lines= ax.plot(x_data,prediction_value,'r-',lw=5)   #划线
        plt.pause(0.1)                                      #暂停0.1秒

