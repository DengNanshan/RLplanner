#log##############

# 1 为什么没有收敛？？lost一直变少？(归一化没做好)
#2 发现之前的加速度都是0，根本没有更新（已经修改）

# 3  前面车的策略要改，要不仅仅后车变化，前车也要变化
import matplotlib as mpl
import xlsxwriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LogNorm
from matplotlib.pyplot import *
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
from IDM import IDM
from Head_vehicle import  Head_vehicle
from keras.utils import to_categorical
# FV = Head_vehicle()
# FV.paraacc =0.3
# EGO = IDM()
# FV.decision_dt = 0.1
# EGO.decision_dt = 0.1
# vehicle_num=20
# if FV.decision_dt == EGO.decision_dt:
#     print('1')
# else:
#     print('false')
# t = [] # 时间
# fs = []  # 前车距离
# fv = []   # 前车速度
# s = []    # 距离
# v = []    # 速度
# a = []    # 自身加速度
# lable=[]
# i = 0
#
# num_classes=10  # 这里的是多少种地方
# for v_num in range(1,vehicle_num):
#     EGO.reset()
#     FV.reset()
#     print('generate vehicle data',v_num)
#     samp = np.random.randn(10)
#     for dTTT in range(1,10):
#         EGO.T = dTTT
#         for sample in samp:
#             EGO.T=(dTTT+sample*0.3)
#             while FV.position < dTTT*1000+sample*100:
#                 i = i + 1
#                 FV.update_state()
#                 EGO.update_state(FV.position, FV.speed)
#                 t.append(i * EGO.decision_dt)
#                 fs.append(FV.position)
#                 fv.append(FV.speed)
#                 s.append(EGO.ego_position)
#                 v.append(EGO.ego_speed)
#                 a.append(EGO.ego_accl)
#                 lable.append(dTTT)
# #np.save('data'+time.strftime('%Y-%m-%d %H:%M:%S', t3),a)
# # plt.figure(1)
# # f1 = plt.plot(t, s)
# # f2 = plt.plot(t, fs)
# # plt.legend(labels=['s', 'fs'])
# # plt.figure(2)
# ds=np.array(fs)-np.array(s)
# # fds = plt.plot(t, ds)
# # plt.show()
#
# #加载数据
# #归一化
# ds=np.array(ds)
# v=np.array(v)
# a=np.array(a)
# fv=np.array(fv)
#
# # dsc=(ds-min(ds))/(max(ds)-min(ds))
# # vc=(v-min(v))/(max(v)-min(v))
# # ac=(a-min(a))/(max(a)-min(a))
# # fvc=(fv-min(fv))/(max(fv)-min(fv))
#
# dsc=ds/100
# vc=v/50
# ac=(a+2)/4
# fvc=fv/50
#
# x_train=np.array([dsc,
#          vc,
#          ac,
#          fvc])
# x_train=x_train.T
#
# lable=np.array(lable)
# y_test = to_categorical(lable,0 )
# #打乱数据
# index = [i for i in range(len(x_train))]
# np.random.shuffle(index)
# x_train = x_train[index]
# lable = lable[index]
# y_test=y_test[index]
##########          1  2  3  4  5  6  7  8  9
traindata=np.array([1, 1, 1, 1, 0, 1, 1])
testdata= np.array([1, 1, 1, 1, 1, 1, 1])
df1 = pd.read_csv('D:\\doctor project\\2019VAE\\数据库\\highD-dataset-v1.0\\postiondel01.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
df2 = pd.read_csv('D:\\doctor project\\2019VAE\\数据库\\highD-dataset-v1.0\\postiondel05.csv')
df3 = pd.read_csv('D:\\doctor project\\2019VAE\\数据库\\highD-dataset-v1.0\\postiondel15.csv')
df4 = pd.read_csv('D:\\doctor project\\2019VAE\\数据库\\US-101-LosAngeles-CA\\postiondelus101.csv')
df5 = pd.read_csv('D:\\doctor project\\2019VAE\\数据库\\I-80-Emeryville-CA\\postiondeli80.csv')
df6 = pd.read_csv('D:\\doctor project\\2019VAE\\数据库\\Peachtree-Street-Atlanta-GA\\postiondelPeachdata.csv')
df7 = pd.read_csv('D:\\doctor project\\2019VAE\\数据库\\Lankershim-Boulevard-LosAngeles-CA\\postiondellink.csv')

df1.columns = ["Ve", "Vh", "a", "dS"]
df2.columns = ["Ve", "Vh", "a", "dS"]
df3.columns = ["Ve", "Vh", "a", "dS"]
df4.columns = ["Ve", "Vh", "a", "dS"]
df5.columns = ["Ve", "Vh", "a", "dS"]
df6.columns = ["Ve", "Vh", "a", "dS"]
df7.columns = ["Ve", "Vh", "a", "dS"]

X1 = df1[["dS", "Ve", "a", "Vh"]]  # 抽取前七列作为训练数据的各属性值
X2 = df2[["dS", "Ve", "a", "Vh"]]  # 抽取前七列作为训练数据的各属性值
X3 = df3[["dS", "Ve", "a", "Vh"]]  # 抽取前七列作为训练数据的各属性值
X4 = df4[["dS", "Ve", "a", "Vh"]]  # 抽取前七列作为训练数据的各属性值
X5 = df5[["dS", "Ve", "a", "Vh"]]  # 抽取前七列作为训练数据的各属性值
X6 = df5[["dS", "Ve", "a", "Vh"]]  # 抽取前七列作为训练数据的各属性值
X7 = df5[["dS", "Ve", "a", "Vh"]]  # 抽取前七列作为训练数据的各属性值
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
X5 = np.array(X5)
X6 = np.array(X6)
X7 = np.array(X7)
X4=X4[1:10000]
X5=X5[1:10000]
X6=X6[1:10000]
X7=X7[1:10000]
alldata=np.array([X1,X2,X3,X4,X5,X6,X7])

# b=[]
# for i in range(len(traindata)):
#     if traindata[i] ==1:
#         b.append(i)
# X=np.array([[ , , ,]])
# for i in range(len(b)):

X=np.vstack(( X3,X4 ))

# X[:,0]=X[:,0]/100
# X[:,1]=X[:,1]/50
# X[:,2]=(X[:,2]+2)/4
# X[:,3]=X[:,3]/50


X[:,0]=(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0]))
X[:,1]=(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1]))
X[:,2]=(X[:,2]-min(X[:,2]))/(max(X[:,2])-min(X[:,2]))
X[:,3]=(X[:,3]-min(X[:,3]))/(max(X[:,3])-min(X[:,3]))
x_train=X

y1=np.ones(len(X1))*1
y2=np.ones(len(X2))*1
y3=np.ones(len(X3))*1
y4=np.ones(len(X4))*2
y5=np.ones(len(X5))*3
y6=np.ones(len(X6))*4
y7=np.ones(len(X7))*5
lable=np.hstack((y3,y4))
lable_false=np.ones(len(lable))*5 #(这里还做了一个修改，loss没有计算这部分)
y_test = to_categorical(lable_false,0 )


index = [i for i in range(len(x_train))]
np.random.shuffle(index)
x_train = x_train[index]
lable = lable[index]
y_test=y_test[index]

batch_size = 100
original_dim = 4
latent_dim = 2
intermediate_dim = 4
epochs = 2


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

y = Input(shape=(6,))
yh = Dense(latent_dim)(y)

# 重参数


def sampling(args):
    z_mean1, z_log_var1 = args
    epsilon = K.random_normal(shape=K.shape(z_mean1))
    return z_mean + K.exp(z_log_var1 / 2) * epsilon


# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')  ##  这里是不是应该变成0-1
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# 建立模型
# vae = Model(x, x_decoded_mean)
vae = Model([x,y],[x_decoded_mean,yh])

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
#一个较高的beta值，就使得前变量空间z表示信息的丰富度降低，但同时模型的解纠缠能力增加。所以beta可以作为表示能力和解纠缠能力之间的平衡因子。

# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐

kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean-yh) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss +kl_loss)
# K.categorical_crossentropy()
# K.binary_crossentropy()

# K.sparse_categorical_crossentropy()
# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

vae.fit([x_train, y_test],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_train, y_test], None))


# 构建encoder，然后观察各个在隐空间的分布
encoder = Model(x, z_mean)
encoder2 = Model(x, z_log_var)
# # 输出每个类的均值向量
# mu = Model(y, yh)
# mu = mu.predict(np.eye(6))



x_test_encoded = encoder.predict(x_train, batch_size=batch_size)
x_test_encoded1 = encoder.predict(X, batch_size=batch_size)
x_test_encoded2 = encoder2.predict(x_train, batch_size=batch_size)





###################################################plot
plt.figure(figsize=(6, 6))
discount=2
xt=x_test_encoded[::discount]
lable1=lable[::discount]
# ss=s[::100]
# tt=t[::100]
plt.figure(1)
plt.scatter(xt[:, 0], xt[:, 1], c=lable1)
# plt.figure(5)
# plt.scatter(x_test_encoded1[:, 0], x_test_encoded1[:, 1])
# plt.scatter(xt[:, 0], lable1, c=lable1)
# plt.scatter(xt[:, 1], lable1, c=lable1)
# xttt=xt[:,0]*xt[:, 0]+xt[:, 0]*xt[:, 0]
# xttt=pow(xttt,0.5)
# plt.plot(ss,xttt)
# plt.plot(ss,lable1)
plt.colorbar()



# plt.figure(2)
# b=plt.hist(xt[:, 1],discount)

##  butongya

# # aaaa=lable1.tolist()
# # j=1
b1=[]
b2=[]
b3=[]
b4=[]
b5=[]
b6=[]
b7=[]
b8=[]
for i in range(1,len(lable1)):
    if lable1[i] == 3:
        b3.append(xt[i])
    elif lable1[i] == 1:
        b1.append(xt[i])
    elif lable1[i] == 2:
        b2.append(xt[i])
    elif lable1[i]==4:
        b4.append(xt[i])
    elif lable1[i] == 5:
        b5.append(xt[i])
    elif lable1[i] == 6:
        b6.append(xt[i])
    elif lable1[i] == 7:
        b7.append(xt[i])
    elif lable1[i] == 8:
        b8.append(xt[i])

b1=np.array(b1)
b2=np.array(b2)
b3=np.array(b3)
b4=np.array(b4)
b5=np.array(b5)
b6=np.array(b6)
b7=np.array(b7)




# b2=aaaa.index(2,3,10)
# b3=aaaa.index(3)
# b4=aaaa.index(4)
# plt.figure(5)

# plt.hist(b3,100)
# plt.hist(b4,100)
# plt.hist(b5,100)
# plt.hist(b6,100)
# plt.hist(b7,100)


mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(3)
c=plt.hist2d(xt[:, 0],xt[:, 1],30)
plt.colorbar()

X11, Y11 = np.meshgrid(c[1][0:30], c[2][0:30])
fig = plt.figure(4)
ax = fig.gca(projection='3d')
Z=c[0]
Axes3D.plot_surface(ax,X11,Y11,Z)

plt.figure(5)
plt.title("决策的地理特异性提取结果")
scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
plt.xlabel("第一维度 Z1")
plt.ylabel("第一维度 Z1")
x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2,
                               sharex=scatter_axes)
plt.title("第一维度分布")
plt.xlabel("第一维度 Z1")
plt.ylabel("数量")

y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2,
                               sharey=scatter_axes)
plt.title("第二维度分布")
plt.xlabel("数量")
plt.ylabel("第二维度 Z2")
# scatter_axes.plot(xt[:, 0], xt[:, 1], '.')
# x_hist_axes.hist(xt[:, 0],100)
# y_hist_axes.hist(xt[:, 1],100, orientation='horizontal')
#


scatter_axes.plot(b2[:, 0], b2[:, 1], '.')
x_hist_axes.hist(b2[:, 0],100)
y_hist_axes.hist(b2[:, 1],100, orientation='horizontal')

# scatter_axes.plot(b3[:, 0], b3[:, 1], '.')
# x_hist_axes.hist(b3[:, 0],100)
# y_hist_axes.hist(b3[:, 1],100, orientation='horizontal')

# scatter_axes.plot(b4[:, 0], b4[:, 1], '.')
# x_hist_axes.hist(b4[:, 0],100)
# y_hist_axes.hist(b4[:, 1],100, orientation='horizontal')

scatter_axes.plot(b1[:, 0], b1[:, 1], '.')
x_hist_axes.hist(b1[:, 0],100)
y_hist_axes.hist(b1[:, 1],100, orientation='horizontal')

# scatter_axes.plot(b5[:, 0], b5[:, 1], '.')
# x_hist_axes.hist(b5[:, 0],100)
# y_hist_axes.hist(b5[:, 1],100, orientation='horizontal')
plt.legend("43")
# scatter_axes.plot(b5[:, 0], b5[:, 1], '.')
# x_hist_axes.hist(b5[:, 0],100)
# y_hist_axes.hist(b5[:, 1],100, orientation='horizontal')
# vae.save('vae1217-20-32.h5')
# encoder.save('encoder1217-20-32.h5')
# a=load_model('firstmodel.h5')



plt.show()


############KLsandu
import numpy as np
import scipy.stats
p=np.asarray([0.65,0.25,0.07,0.03])
q=np.array([0.6,0.25,0.1,0.05])
def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)
print(KL_divergence(p,q)) # 0.011735745199107783
print(KL_divergence(q,p)) # 0.013183150978050884

# plt.hist()
# x, bins=10, range=None, normed=False,
#     weights=None, cumulative=False, bottom=None,
#     histtype=u'bar', align=u'mid', orientation=u'vertical',
#     rwidth=None, log=False, color=None, label=None, stacked=False,
#     hold=None, **kwargs) p
# plt.hist(a,bins=10,range=[1,3])
print(KL_divergence(p,q))