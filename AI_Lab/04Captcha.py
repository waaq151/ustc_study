
from captcha.image import ImageCaptcha # pip install captcha 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np
import tensorflow as tf


number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = []
ALPHABET = []

##生成n位验证码 这里n=4
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for _ in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


#生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text() 
    captcha_text = ''.join(captcha_text) #连接字符串

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha) 
    captcha_image = np.array(captcha_image) 
    return captcha_text, captcha_image

# 显示验证码图片
def show_image(text,image):
    ax = plt.figure().add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()


# 图像大小
IMAGE_HEIGHT = 60   # 图像高
IMAGE_WIDTH = 160   # 图像宽
MAX_CAPTCHA = 4     # 图中验证码数目
print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(image):
    image = Image.fromarray(image)
    image = image.convert('L') 
    return np.asarray(image)
# 测试convert2gray
# text, image = gen_captcha_text_and_image()
# print("验证码图像channel:", image.shape)  # (60, 160, 3)
# image = convert2gray(image)
# print("验证码图像channel:", image.shape)  # (60, 160)


# 文本转向量(使用字符在CHAR_SET中的下标作为标签)
CHAR_SET = number + alphabet + ALPHABET
def text2vec(text):
	if len(text) > MAX_CAPTCHA:
		raise ValueError('验证码最长4个字符')

	vector = np.zeros(MAX_CAPTCHA*len(CHAR_SET))
	for i, c in enumerate(text):
		idx = i * len(CHAR_SET) + CHAR_SET.index(c)
		vector[idx] = 1
	return vector


# 向量转回文本
def vec2text(vec):
    text = str()
    for id, c in enumerate(vec):
        if abs(c -1) < 0.001:
            char_id = id % len(CHAR_SET)
            text += CHAR_SET[char_id]
    return text

# 测试文本、向量之间的转换
# vec = text2vec("10")
# text = vec2text(vec)
# print(text)  
# vec = text2vec("01")
# text = vec2text(vec)
# print(text)  


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size,IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*len(CHAR_SET)])

    # 过滤图像大小不是(60, 160, 3)的
    def wrap_gen_captcha_text_and_image():
	    while True:
		    text, image = gen_captcha_text_and_image()
		    if image.shape == (60, 160, 3):
			    return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0  flatten将对象转化为一维
        batch_y[i,:] = text2vec(text)
    return batch_x, batch_y


# 权重初始化
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 偏置初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 网络模型
def model(X, p_keep_conv, p_keep_hidden):
    # 卷积核（权重矩阵）
    w = init_weights([3, 3, 1, 32])         # 3x3x1 conv, 32 outputs    3×3是patch的大小，即从输入中提取的图块的尺寸，1是输入的通道数目，32是输出的通道数目，即卷积所计算的过滤器数量
    w2 = init_weights([3, 3, 32, 64])       # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 64])       # 3x3x32 conv, 64 outputs
    w4 = init_weights([64 * 20 * 8, 1024])  # FC 64 * 20 * 8 inputs, 1024 outputs
    w_o = init_weights([1024, MAX_CAPTCHA*len(CHAR_SET)])         # FC 1024 inputs, 40 outputs (labels)
    # 偏置
    b = bias_variable([32])
    b2 = bias_variable([64])
    b3 = bias_variable([64])
    b4 = bias_variable([1024])
    b_o = bias_variable([MAX_CAPTCHA*len(CHAR_SET)])

    with tf.name_scope('layer1'):
        l1a = tf.nn.relu(conv2d(X, w) + b)                      # l1a shape=(?, 60, 160, 32)     卷积是线性操作，通过非线性激活函数relu来扩展假设空间   relu函数将所有负值归零   
        l1 = max_pool(l1a)                                      # l1 shape=(?, 30, 80, 32)      池化层往往在卷积层后面，通过池化来降低卷积层输出的特征向量，同时改善结果（不易出现过拟合）。           
        l1 = tf.nn.dropout(l1, p_keep_conv)                     # dropout可以屏蔽神经元的输出，减少过拟合。p_keep_conv表示dropout中保持不变的概率。
    with tf.name_scope('layer2'):
        l2a = tf.nn.relu(conv2d(l1, w2) + b2)                   # l2a shape=(?, 30, 80, 64)    
        l2 = max_pool(l2a)                                      # l2 shape=(?, 15, 40, 64) 
        l2 = tf.nn.dropout(l2, p_keep_conv)
    with tf.name_scope('layer3'):
        l3a = tf.nn.relu(conv2d(l2, w3) + b3)                    # l3a shape=(?, 15, 40, 64)
        l3 = max_pool(l3a)                                        # l3 shape=(?, 8, 20, 64)
        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 10240)        8*20*64=10240  将l3平铺到1维张量
        l3 = tf.nn.dropout(l3, p_keep_conv)
    with tf.name_scope('layer4'):
        l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)                        # 隐藏层，输出1024个隐藏单元
        l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o) + b_o                  
    return pyx


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*len(CHAR_SET)])
p_keep_conv = tf.placeholder("float")       # 卷积层dropout中不变的概率
p_keep_hidden = tf.placeholder("float")     # 隐藏层dropout中不变的概率
BATCH_SIZE = 128 

# 训练
def train_crack_captcha_cnn():
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    py_x = model(x, p_keep_conv, p_keep_hidden)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=py_x, labels=Y))   # 对于多标签、多分类问题，网络的最后一层应该使用sigmoid 激活
        train_op = tf.train.AdamOptimizer(0.001).minimize(cost)                         # 使用RMSProp优化器构建模型
        tf.summary.scalar('cost',cost)  # 记录标量数据

    with tf.name_scope('accuracy'):
        # correct_pred = tf.equal(tf.argmax(py_x,1),tf.argmax(Y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        # tf.summary.scalar('accuracy', accuracy)
        predict = tf.reshape(py_x, [-1, MAX_CAPTCHA, len(CHAR_SET)])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, len(CHAR_SET)]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy',accuracy)  # 记录标量数据
        
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./AI_Lab/logs/captcha1_logs", sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(128)
            _, loss = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y, p_keep_conv: 0.75, p_keep_hidden:0.75})
            print(step, loss)
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                summary,acc = sess.run([merged, accuracy], feed_dict={X:batch_x_test, Y:batch_y_test, p_keep_conv:1.0, p_keep_hidden:1.0})
                writer.add_summary(summary,step)
                print('准确率:',step, acc)
                # 如果准确率大于99%,保存模型,完成训练
                if acc > 0.99:
                    saver.save(sess, "./AI_Lab/04model1/crack_capcha.model", global_step=step)
                    break
                
            step += 1


train_crack_captcha_cnn()


def crack_captcha(captcha_image):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    test_model = model(x, p_keep_conv, p_keep_hidden)
 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./AI_Lab/04model'))

        predict = tf.argmax(tf.reshape(test_model, [-1, MAX_CAPTCHA, len(CHAR_SET)]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], p_keep_conv:1.0, p_keep_hidden:1.0})
        text = text_list[0].tolist()
        return text
 
text, image = gen_captcha_text_and_image()
image = convert2gray(image)
image = image.flatten() / 255
predict_text = crack_captcha(image)
print("正确: {}  预测: {}".format(text, predict_text))