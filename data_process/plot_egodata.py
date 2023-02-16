
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

file_path = "../Data/egodata/"
file_name = "case2-1"

print(file_path+file_name+".csv")
df = pd.read_csv(file_path+file_name+".csv")
df_env = pd.read_csv(file_path+file_name + "env.csv")

# 创建3行2列的子图布局
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 8))
x = df["x"]
y = df["y"]
time = df["timestamp"] - df["timestamp"][0]
speed = df["speed_mps"]
accx = df["accx"]
accy = df["accy"]
df["acc"] = (df["accx"]**2+ df["accy"]**2)**0.5

steering = df["steering_percentage"]

print(accx)

# 创建GridSpec布局，将右侧的子图设置为占据2个位置
gs = GridSpec(3, 2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[2, 0])
ax4 = plt.subplot(gs[:, 1])
# ax4 = axs[:,1]

# 绘制子图
ax1.plot(accx )
ax1.plot(accy )
ax1.set_title('Acc')
ax1.set_xlabel("time t(s)")



ax3.plot(time,steering)
ax3.set_xlabel("time t(s)")
ax3.set_ylabel("steering")

ax2.plot(time,speed*3.6)
ax2.set_xlabel("time t(s)")
ax2.set_title('speed')

cax=ax4.scatter(x,y, c=time,cmap="viridis",label = "ego")
ax4.scatter(df_env["x"],df_env["y"], s = 1,c=df_env["timestamp"],cmap="viridis",label = "env")
ax4.set_xlabel("UTM x(m)")
ax4.set_ylabel("UTM y(m)")
ax4.set_title('Trajectory of ego vehicle')
ax4.axis('equal')
ax4.legend()


cbar=fig.colorbar(cax, ax=ax4)
cbar.outline.set_linewidth(0.1)



# fig.colorbar(ax4, ax=axs[:,1])
plt.tight_layout()
plt.savefig(file_path+file_name+".png",transparent =False)
plt.savefig(file_path+file_name+"_t.png",transparent =True)
plt.show()


