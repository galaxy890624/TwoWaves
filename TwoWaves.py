
#by galaxy890624

from xmlrpc.client import FastParser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 創建數據
data = np.zeros(5)
data[0] = input("a_wave1 = ")  # a_wave1 = data[0]
data[1] = input("a_wave2 = ")  # a_wave2 = data[1]
data[2] = input("h = ")  # h = data[2]
data[3] = input("T_wave1 = ")  # T_wave1 = data[3]
data[4] = input("T_wave2 = ")  # T_wave2 = data[4]

fps = 25
t = [i/fps for i in range(0, 12*fps, 1)]  # 使用列表推導式生成 0~12 秒，共 300 個數據點
g = 9.8
sigma_wave1 = 2 * np.pi / data[3]  # print(sigma) # 1.0471975511965976
sigma_wave2 = 2 * np.pi / data[4]  # print(sigma) # 1.0471975511965976
frame = 0  # 初始化

k_data_wave1 = np.zeros(21*2).reshape(21, 2)
k_data_wave1[0][1] = 1

k_data_wave2 = np.zeros(21*2).reshape(21, 2)
k_data_wave2[0][1] = 1

for i in range(0, 21, 1):  # i = 0~20
    k_data_wave1[i][0] = i
    k_data_wave1[i][1] = pow(sigma_wave1, 2) / (g * np.tanh(k_data_wave1[i-1][1] * data[2]))
    print("iter =", i, ", k_wave1 =", k_data_wave1[i][1])

for i in range(0, 21, 1):  # i = 0~20
    k_data_wave2[i][0] = i
    k_data_wave2[i][1] = pow(sigma_wave2, 2) / (g * np.tanh(k_data_wave2[i-1][1] * data[2]))
    print("iter =", i, ", k_wave2 =", k_data_wave2[i][1])

theta_wave1 = np.deg2rad(float(input("theta_wave1 in degree = ")))
theta_wave2 = np.deg2rad(float(input("theta_wave2 in degree = ")))

kx_1 = k_data_wave1[20][1] * np.cos(theta_wave1)
ky_1 = k_data_wave1[20][1] * np.sin(theta_wave1)

kx_2 = k_data_wave2[20][1] * np.cos(theta_wave2)
ky_2 = k_data_wave2[20][1] * np.sin(theta_wave2)


# 初始化圖形_wave1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化
def eta_wave1(x, y, frame):
    return data[0] * np.cos(kx_1 * x + ky_1 * y - sigma_wave1 * t[int(frame)])

def eta_wave2(x, y, frame):
    return data[1] * np.cos(kx_2 * x + ky_2 * y - sigma_wave2 * t[int(frame)])

def eta_total(x, y, frame):
    return eta_wave1(x, y, frame) + eta_wave2(x, y, frame)

# 定義更新函數
def update_wave1(frame):
    # 根據當前幀的索引來獲取對應的數據值
    x = np.linspace(0, 250, 51)
    y = np.linspace(0, 250, 51)
    X, Y = np.meshgrid(x, y)

    ax.cla()  # 清除軸上的內容
    ax.set_xlabel("x(meter)")
    ax.set_ylabel("y(meter)")
    ax.set_zlabel("z(meter)")
    ax.set_title("eta_wave1")

    # 更新曲面數據
    Z = eta_wave1(X, Y, frame)
    ax.plot_surface(X, Y, Z, cmap='viridis')

# 創建動畫
ani_wave1 = FuncAnimation(fig, update_wave1, frames=len(t), interval=1000/fps)
# 請注意，我們使用 interval=40 來設置每幀的間隔時間為 40 毫秒（0.04 秒）。這將確保動畫在正確的時間節奏下播放。

#顯示動畫
plt.show()


# 初始化圖形_wave2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化
def eta_wave1(x, y, frame):
    return data[0] * np.cos(kx_1 * x + ky_1 * y - sigma_wave1 * t[int(frame)])

def eta_wave2(x, y, frame):
    return data[0] * np.cos(kx_2 * x + ky_2 * y - sigma_wave2 * t[int(frame)])

def eta_total(x, y, frame):
    return eta_wave1(x, y, frame) + eta_wave2(x, y, frame)

# 定義更新函數
def update_wave2(frame):
    # 根據當前幀的索引來獲取對應的數據值
    x = np.linspace(0, 250, 51)
    y = np.linspace(0, 250, 51)
    X, Y = np.meshgrid(x, y)

    ax.cla()  # 清除軸上的內容
    ax.set_xlabel("x(meter)")
    ax.set_ylabel("y(meter)")
    ax.set_zlabel("z(meter)")
    ax.set_title("eta_wave2")

    # 更新曲面數據
    Z = eta_wave2(X, Y, frame)
    ax.plot_surface(X, Y, Z, cmap='viridis')

# 創建動畫
ani_wave1 = FuncAnimation(fig, update_wave2, frames=len(t), interval=1000/fps)
# 請注意，我們使用 interval=40 來設置每幀的間隔時間為 40 毫秒（0.04 秒）。這將確保動畫在正確的時間節奏下播放。

#顯示動畫
plt.show()


# 初始化圖形_total
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化
def eta_wave1(x, y, frame):
    return data[0] * np.cos(kx_1 * x + ky_1 * y - sigma_wave1 * t[int(frame)])

def eta_wave2(x, y, frame):
    return data[0] * np.cos(kx_2 * x + ky_2 * y - sigma_wave2 * t[int(frame)])

def eta_total(x, y, frame):
    return eta_wave1(x, y, frame) + eta_wave2(x, y, frame)

# 定義更新函數
def update_total(frame):
    # 根據當前幀的索引來獲取對應的數據值
    x = np.linspace(0, 250, 51)
    y = np.linspace(0, 250, 51)
    X, Y = np.meshgrid(x, y)

    ax.cla()  # 清除軸上的內容
    ax.set_xlabel("x(meter)")
    ax.set_ylabel("y(meter)")
    ax.set_zlabel("z(meter)")
    ax.set_title("eta_total")

    # 更新曲面數據
    Z = eta_total(X, Y, frame)
    ax.plot_surface(X, Y, Z, cmap='viridis')

# 創建動畫
ani_wave1 = FuncAnimation(fig, update_total, frames=len(t), interval=1000/fps)
# 請注意，我們使用 interval=40 來設置每幀的間隔時間為 40 毫秒（0.04 秒）。這將確保動畫在正確的時間節奏下播放。

#顯示動畫
plt.show()
