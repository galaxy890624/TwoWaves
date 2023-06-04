
#by galaxy890624

from xmlrpc.client import FastParser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# �Ыؼƾ�
data = np.zeros(5)
data[0] = input("a_wave1 = ")  # a_wave1 = data[0]
data[1] = input("a_wave2 = ")  # a_wave2 = data[1]
data[2] = input("h = ")  # h = data[2]
data[3] = input("T_wave1 = ")  # T_wave1 = data[3]
data[4] = input("T_wave2 = ")  # T_wave2 = data[4]

fps = 25
t = [i/fps for i in range(0, 12*fps, 1)]  # �ϥΦC����ɦ��ͦ� 0~12 ��A�@ 300 �Ӽƾ��I
g = 9.8
sigma_wave1 = 2 * np.pi / data[3]  # print(sigma) # 1.0471975511965976
sigma_wave2 = 2 * np.pi / data[4]  # print(sigma) # 1.0471975511965976
frame = 0  # ��l��

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


# ��l�ƹϧ�_wave1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ��l��
def eta_wave1(x, y, frame):
    return data[0] * np.cos(kx_1 * x + ky_1 * y - sigma_wave1 * t[int(frame)])

def eta_wave2(x, y, frame):
    return data[1] * np.cos(kx_2 * x + ky_2 * y - sigma_wave2 * t[int(frame)])

def eta_total(x, y, frame):
    return eta_wave1(x, y, frame) + eta_wave2(x, y, frame)

# �w�q��s���
def update_wave1(frame):
    # �ھڷ�e�V�����ި�����������ƾڭ�
    x = np.linspace(0, 250, 51)
    y = np.linspace(0, 250, 51)
    X, Y = np.meshgrid(x, y)

    ax.cla()  # �M���b�W�����e
    ax.set_xlabel("x(meter)")
    ax.set_ylabel("y(meter)")
    ax.set_zlabel("z(meter)")
    ax.set_title("eta_wave1")

    # ��s�����ƾ�
    Z = eta_wave1(X, Y, frame)
    ax.plot_surface(X, Y, Z, cmap='viridis')

# �Ыذʵe
ani_wave1 = FuncAnimation(fig, update_wave1, frames=len(t), interval=1000/fps)
# �Ъ`�N�A�ڭ̨ϥ� interval=40 �ӳ]�m�C�V�����j�ɶ��� 40 �@��]0.04 ��^�C�o�N�T�O�ʵe�b���T���ɶ��`���U����C

#��ܰʵe
plt.show()


# ��l�ƹϧ�_wave2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ��l��
def eta_wave1(x, y, frame):
    return data[0] * np.cos(kx_1 * x + ky_1 * y - sigma_wave1 * t[int(frame)])

def eta_wave2(x, y, frame):
    return data[0] * np.cos(kx_2 * x + ky_2 * y - sigma_wave2 * t[int(frame)])

def eta_total(x, y, frame):
    return eta_wave1(x, y, frame) + eta_wave2(x, y, frame)

# �w�q��s���
def update_wave2(frame):
    # �ھڷ�e�V�����ި�����������ƾڭ�
    x = np.linspace(0, 250, 51)
    y = np.linspace(0, 250, 51)
    X, Y = np.meshgrid(x, y)

    ax.cla()  # �M���b�W�����e
    ax.set_xlabel("x(meter)")
    ax.set_ylabel("y(meter)")
    ax.set_zlabel("z(meter)")
    ax.set_title("eta_wave2")

    # ��s�����ƾ�
    Z = eta_wave2(X, Y, frame)
    ax.plot_surface(X, Y, Z, cmap='viridis')

# �Ыذʵe
ani_wave1 = FuncAnimation(fig, update_wave2, frames=len(t), interval=1000/fps)
# �Ъ`�N�A�ڭ̨ϥ� interval=40 �ӳ]�m�C�V�����j�ɶ��� 40 �@��]0.04 ��^�C�o�N�T�O�ʵe�b���T���ɶ��`���U����C

#��ܰʵe
plt.show()


# ��l�ƹϧ�_total
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ��l��
def eta_wave1(x, y, frame):
    return data[0] * np.cos(kx_1 * x + ky_1 * y - sigma_wave1 * t[int(frame)])

def eta_wave2(x, y, frame):
    return data[0] * np.cos(kx_2 * x + ky_2 * y - sigma_wave2 * t[int(frame)])

def eta_total(x, y, frame):
    return eta_wave1(x, y, frame) + eta_wave2(x, y, frame)

# �w�q��s���
def update_total(frame):
    # �ھڷ�e�V�����ި�����������ƾڭ�
    x = np.linspace(0, 250, 51)
    y = np.linspace(0, 250, 51)
    X, Y = np.meshgrid(x, y)

    ax.cla()  # �M���b�W�����e
    ax.set_xlabel("x(meter)")
    ax.set_ylabel("y(meter)")
    ax.set_zlabel("z(meter)")
    ax.set_title("eta_total")

    # ��s�����ƾ�
    Z = eta_total(X, Y, frame)
    ax.plot_surface(X, Y, Z, cmap='viridis')

# �Ыذʵe
ani_wave1 = FuncAnimation(fig, update_total, frames=len(t), interval=1000/fps)
# �Ъ`�N�A�ڭ̨ϥ� interval=40 �ӳ]�m�C�V�����j�ɶ��� 40 �@��]0.04 ��^�C�o�N�T�O�ʵe�b���T���ɶ��`���U����C

#��ܰʵe
plt.show()
