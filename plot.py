import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设X是一个2xn维的矩阵
X = np.random.rand(2, 100)  # 生成随机数据作为示例

# 假设y_pred和y也是相应的3维数据
y_pred = np.random.rand(1, 100)
y = np.random.rand(1, 100)

# 创建一个新的三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制预测值和真实值
ax.scatter(X[0], X[1], y_pred, color='b', label='Predicted')
ax.scatter(X[0], X[1], y, color='r', label='Actual')

# 设置图例和标签
ax.legend()
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# 显示图像
plt.show()
