import matplotlib.pyplot as plt
import pandas as pd


file_path = "../Data/routing/"
file_name = "garage4_length"
df = pd.read_csv(file_path+file_name+".csv")


x = df["x"]
y = df["y"]

plt.scatter(x,y, c=df["time"],cmap="viridis")

plt.axis('equal')
plt.xlabel("UTM x(m)")
plt.ylabel("UTM y(m)")
plt.title(file_name)
plt.colorbar()
plt.savefig(file_path+file_name+".png")

plt.show()
