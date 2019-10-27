import tensorflow as tf

# 导入数据集
from tensorflow.examples.tutorials.mnist import input_data
# one_hot 独热编码，也叫一位有效编码。在任意时候只有一位为1，其他位都是0
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels
# 一批的数量
batch_size = 100

#训练的x(image),y(label)
x = tf.placeholder(tf.float32, [None, 784]) 
y = tf.placeholder(tf.float32, [None, 10])

# W模型权重
#[55000,784] * W = [55000,10]
W = tf.Variable(tf.zeros([784, 10]))  #创建一个所有元素都设置为零的张量。784 = 28*28

# b模型的偏置量/干扰量  由于输入往往会带一些无关的干扰量，需要再加上一个额外的偏置量b
b = tf.Variable(tf.zeros([10]))

# 用softmax构建逻辑回归模型（把证据转化为N个输出类别上的概率分布）多分类、单标签问题一般用softmax作为激活函数
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数(交叉熵)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), 1)) #概率越大，信息量越少

# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# 加载session图
with tf.Session() as sess:
    sess.run(init)

    # 开始训练
    for epoch in range(25): #  1个epoch等于使用训练集中的全部样本训练一次
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 按批训练batch_size = 100
            sess.run(optimizer, {x: batch_xs,y: batch_ys})
            #计算损失平均值
            avg_cost += sess.run(cost,{x: batch_xs,y: batch_ys}) / total_batch
        if (epoch+1) % 5 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("运行完成")

    # 测试求正确率
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print("正确率:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))