
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

file_path = "../Data/egodata/"
file_name = "case1-1"

print(file_path+file_name+".csv")
df = pd.read_csv(file_path+file_name+".csv")
df_env = pd.read_csv(file_path+file_name + "env.csv")

# 创建3行2列的子图布局
fig, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(4, 8))
x = df["x"]
y = df["y"]
time = df["timestamp"] - df["timestamp"][0]
speed = df["speed_mps"]
accx = df["accx"]
accy = df["accy"]
df["acc"] = (df["accx"]**2+ df["accy"]**2)**0.5

steering = df["steering_percentage"]


# 创建GridSpec布局，将右侧的子图设置为占据2个位置
# gs = GridSpec(3, 2)


# ax4 = axs[:,1]

# 绘制子图


cax=ax4.scatter(x,y, c=time,cmap="viridis",label = "ego",s = 2)
ax4.scatter(df_env["x"],df_env["y"], s = 1,c=df_env["timestamp"],cmap="viridis",label = "env",marker = '^')
ax4.set_xlabel("UTM x(m)")
ax4.set_ylabel("UTM y(m)")
ax4.set_title('Trajectory of ego vehicle')
# ax4.axis('equal')
ax4.legend()

if file_name ==  "case1-1":
    ax4.set_xlim(458500, 458600)
    ax4.set_ylim(4399900,4400200)
    # ax4.axis('equal')
    ax4.set_aspect('equal')




cbar=fig.colorbar(cax, ax=ax4)
cbar.outline.set_linewidth(0.1)



# fig.colorbar(ax4, ax=axs[:,1])
plt.tight_layout()
plt.savefig(file_path+file_name+".png",transparent =False)
plt.savefig(file_path+file_name+"_t.png",transparent =True)
plt.show()


