import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128 
test_size = 256

# 权重初始化
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 卷积层（对输入图像进行降维和特征抽取、权值共享，即一个特征图上每个神经元对应的75个权值参数被每个神经元共享
# tf.nn.conv2d(input, filter, strides, padding)
                # input: 需要做卷积的输入图像 [一个batch的图片数量, 高, 宽, 通道数]
                # filter: 卷积核(权重数组) [卷积核的高，卷积核的宽，图像通道数，卷积核个数]
                # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
                # padding: 只能是"SAME","VALID"其中之一，当其为‘SAME’时，表示卷积核可以停留在图像边缘
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层(对输入的特征图进行压缩，一方面使特征图变小，简化网络计算复杂度；一方面进行特征压缩，提取主要特征)
# tf.nn.max_pool(value, ksize, strides, padding, name=None)
                # value: 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map                      依然是[batch, height, width, channels]这样的shape
                # ksize: 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]                              因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                # strides: 和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
                # padding: 和卷积类似，可以取'VALID' 或者'SAME'
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    with tf.name_scope('layer1'):
        l1a = tf.nn.relu(conv2d(X, w))                          # l1a shape=(?, 28, 28, 32)     卷积是线性操作，通过非线性激活函数relu来扩展假设空间   relu函数将所有负值归零   
        l1 = max_pool(l1a)                                      # l1 shape=(?, 14, 14, 32)      池化层往往在卷积层后面，通过池化来降低卷积层输出的特征向量，同时改善结果（不易出现过拟合）。           
        l1 = tf.nn.dropout(l1, p_keep_conv)                     # dropout可以屏蔽神经元的输出，减少过拟合。p_keep_conv表示dropout中保持不变的概率。
    with tf.name_scope('layer2'):
        l2a = tf.nn.relu(conv2d(l1, w2))                        # l2a shape=(?, 14, 14, 64)    
        l2 = max_pool(l2a)                                      # l2 shape=(?, 7, 7, 64) 
        l2 = tf.nn.dropout(l2, p_keep_conv)
    with tf.name_scope('layer3'):
        l3a = tf.nn.relu(conv2d(l2, w3))                        # l3a shape=(?, 7, 7, 128)
        l3 = max_pool(l3a)                                      # l3 shape=(?, 4, 4, 128)
        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)        4*4*128=2048  将l3平铺到1维张量
        l3 = tf.nn.dropout(l3, p_keep_conv)
    with tf.name_scope('layer4'):
        l4 = tf.nn.relu(tf.matmul(l3, w4))                      # 隐藏层，输出625个隐藏单元
        l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)                                  
    return pyx


mnist = input_data.read_data_sets(r".\AI_Lab\MNIST_data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1]) # 把图片变为4d向量，第2、3维对应图片的宽、高，最后1维是颜色通道数
Y = tf.placeholder("float", [None, 10])

# 卷积核（权重矩阵）
w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs    3×3是patch的大小，即从输入中提取的图块的尺寸，1是输入的通道数目，32是输出的通道数目，即卷积所计算的过滤器数量
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)
# 权值的直方图
tf.summary.histogram("w", w)
tf.summary.histogram("w2", w2)
tf.summary.histogram("w3", w3)
tf.summary.histogram("w4", w4)
tf.summary.histogram("w_o", w_o)

p_keep_conv = tf.placeholder("float")       # 卷积层dropout中不变的概率
p_keep_hidden = tf.placeholder("float")     # 隐藏层dropout中不变的概率
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))   # 对于单标签、多分类问题，网络的最后一层应该使用softmax 激活，这样可以输出在N个输出类别上的概率分布(10个类的概率和为1)
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)                         # 使用RMSProp优化器构建模型
    tf.summary.scalar('cost',cost)  # 记录标量数据

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(py_x,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
    tf.summary.scalar('accuracy', accuracy)

#predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("./AI_Lab/logs/cnn_logs", sess.graph)  
    merged = tf.summary.merge_all()

    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(10):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        summary,acc = sess.run([merged, accuracy], feed_dict={X:teX, Y:teY, p_keep_conv:1.0, p_keep_hidden:1.0})
        writer.add_summary(summary,i)
        print(i+1,acc)

        # test_indices = np.arange(len(teX)) # Get A Test Batch
        # np.random.shuffle(test_indices)
        # test_indices = test_indices[0:test_size]

        # print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==sess.run(predict_op, feed_dict={X: teX[test_indices],
        #                                                                                         Y: teY[test_indices],
        #                                                                                         p_keep_conv: 1.0,
        #                                                                                         p_keep_hidden: 1.0})))
