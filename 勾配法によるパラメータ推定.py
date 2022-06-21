
from multiprocessing.connection import wait
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt


#確認問題
#1,#2は別紙参照
#3
X = np.array([ 0.  ,  0.16,  0.22,  0.34,  0.44,  0.5 ,  0.67,  0.73,  0.9 ,  1.  ])
Y = np.array([-0.06,  0.94,  0.97,  0.85,  0.25,  0.09, -0.9 , -0.93, -0.53,  0.08])

X_design_matrix_list=[]
for j in range(len(X)):
    X_element=X[j]
    X_design_matrix=[X_element**i for i in range(len(X))]
    X_design_matrix_list.append(X_design_matrix)
x=np.array(X_design_matrix_list)
# print(x)
# print(x.shape)
xt=x.T
# print(xt)
# print(xt.shape)
y=np.reshape(Y,[-1,1])
# print(Y)
# print(y)
w = np.zeros(x.shape[1])
N=len(x)

max_epochs = 40000
eta0 = 0.03
eps = 1e-4
alpha=10**(-3)

print(alpha)

for t in range(max_epochs):
# for t in range(100):
    eta = eta0 / np.sqrt(t+1)
    i = np.random.randint(0, x.shape[0])
    y_hat = x[i]@w
    y_=y_hat - Y[i]
    # grad = 2 * ((y_hat - y[i]) * X_[i]+ ((alpha/N)*w))
    grad = 2 * ((xt[:,i])*(y_hat - Y[i])+ ((alpha/N)*w))
    #grad = 2* (y_hat - y[i])+ ((alpha/N)*w)
    if np.sum(np.abs(grad)) < eps:
        break
    w -= eta * grad
    # print(w)
    # print('---------------')
    # def sgdLidgeMethod(self, eta=1e-3, eps=1e-4, epoch=50000):
    #     for i in range(epoch):
    #         eta = 0.03/np.sqrt(1+i)
    #         idx = random.randint(0, len(self.sourceX)-1)
    #         grad = self.sgd_lidge(idx)
    #         if np.sum(np.abs(grad)) < eps:
    #             break
    #         self.w -= eta*grad

    # def sgd_lidge(self, idx:int, alpha=1e-6):
    #     return 2*(self.d_matrix.T[:, idx] * (self.d_matrix[idx] @ self.w - self.sourceY[idx]) + alpha *self.w)



#回帰曲線
def Regressioncurve(w,x):
    y=w[0] + w[1] * x + w[2] * (x ** 2) + w[3] * (x ** 3) + w[4] * (x ** 4) + w[5] * (x ** 5) + w[6] * (x ** 6) + w[7] * (x ** 7) + w[8] * (x ** 8) + w[9] * (x ** 9) 
    return y

print(w)
x_graph = np.linspace(np.min(X),np.max(X),1000)
y_graph=Regressioncurve(w,x_graph)


#グラフ描画
Xmin, Xmax = 0, 1
Ymin, Ymax = -1, 1
delta=0.1
plt.figure(figsize=(5,5))
plt.grid(zorder=1)
plt.scatter(X,Y,marker='o',color='b',zorder=2) 
plt.xlabel('x') #x軸のラベル
plt.ylabel('y') #y軸のラベル
plt.xlim(Xmin-delta, Xmax+delta)
plt.ylim(Ymin-delta, Ymax+delta)
plt.plot(x_graph,y_graph,color='m',zorder=3)

plt.show()

# # [ 0.46643393  0.60055031  0.59787968  0.48658218  0.27283108  0.08999875 -0.59189252 -0.83383143 -0.9319528   0.18407473]