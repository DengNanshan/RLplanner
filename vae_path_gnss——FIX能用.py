#! -*- coding: utf-8 -*-
#! -*- coding: utf-8 -*-
'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''
'''用Keras实现的VAE
   目前只保证支持Tensorflow后端
   改写自
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
"""
注意调整bate  resample 是0

？为什么1条轨迹200维也是loss121
日志：原始维度是784的图片压缩为2维，这里尝试吧10Hz的轨迹数据也压缩到2维  
1.数据维度。10Hz的轨迹，10秒钟有100个点。只算xy坐标是200维度。
2.生成方式。x坐标向右，y坐标向上。y为行驶正方向。
3.采样角度为45度到135°。速度恒定为10m/s。
4 尝试增加中间态维度,维度增加到5维度 效果一般
5尝试凸凹正bate resmape0一般
6尝试将输入的轨迹xy分别列出 一般
"""

"""
2023年2月10日01:15:26 基本算是实现了基础功能。需要注意需要防止越界 基本上在0-1之间的数字可以。
2. 
"""


import math
def sample_path(numbers):
    # 生成采样角度45到13度，10s速度恒定为10m/s数据.
    x_train=np.zeros([numbers,200])
    # theta = (np.random.sample(numbers)*90+45)*0.017453292519943295
    theta = (np.random.randn(numbers)*20+90)*0.017453292519943295
    for i in range(numbers):
        c_sin=math.sin(theta[i])
        c_cos=math.cos(theta[i])
        for j in range(100): #100帧率
            #速度采样
            speed_ind =np.random.sample(1)
            # x_train[i][j*2]=(speed_ind*j*c_cos/100  +1)/2 #正则化
            # x_train[i][j*2+1]=speed_ind* j*c_sin/100

            x_train[i][j]=(speed_ind*j*c_cos/100  +1)/2 #正则化
            x_train[i][j+100]=speed_ind* j*c_sin/100
    return x_train

def plot_sample_path(x_train):
    for i in range(x_train.shape[0]):#轨迹数量
        # plt.plot(x_train[i][0:200:2],x_train[i][1:200:2])
        plt.plot(x_train[i][0:100],x_train[i][100:200])



batch_size = 100
original_dim = 200  #28*28
latent_dim = 5 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 256  #16*16
epochs = 100

x_train = y_train_ = x_test = y_test_= sample_path(1000)
plt.figure(1)
plot_sample_path(x_train)
plt.show()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon =  K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon


# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# 建立模型
vae = Model(x, x_decoded_mean)

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)

kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + 0*kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_train, None))


# 构建encoder，然后观察各个数字在隐空间的分布
encoder = Model(x, z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

"""
测试输出
"""



plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])

plt.colorbar()
# plt.show()

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

"""
测试输出
"""
x_decoded = generator.predict(x_test_encoded)
print(x_decoded)
plot_sample_path(x_decoded)
# plt.show()
# 观察隐变量的两个维度变化是如何影响输出结果的
n = 5  # figure with 15x15 digits
digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
plt.figure()
print("plot1")
#用正态分布的分位数来构建隐变量对
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
grid_3 = norm.ppf(np.linspace(0.05, 0.95, n))
grid_4 = norm.ppf(np.linspace(0.05, 0.95, n))
grid_5 = norm.ppf(np.linspace(0.05, 0.95, n))
for i, yi in enumerate(grid_x):
    print()
    for j, xi in enumerate(grid_y):
        for i3, yi3 in enumerate(grid_3):
            for i4 , yi4 in enumerate(grid_4):
                for i5, yi5 in enumerate(grid_5):
                    z_sample = np.array([[xi, yi, yi3, yi4,yi5]])
                    x_decoded = generator.predict(z_sample)
                    plot_sample_path(x_decoded)
                    # digit = x_decoded[0].reshape(digit_size, digit_size)
                    # figure[i * digit_size: (i + 1) * digit_size,
                    #        j * digit_size: (j + 1) * digit_size] = digit

# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
plt.show()
