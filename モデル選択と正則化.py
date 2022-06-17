import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
 
#確認問題

#1
X=np.array([0.,0.16,0.22,0.34,0.44,0.5,0.67,0.73,0.9,1.])
Y=np.array([-0.06,0.94,0.97,0.85,0.25,0.09,-0.9,-0.93,-0.53,0.08])

X_design_matrix_list=[]
for j in range(len(X)):
    X_element=X[j]
    X_design_matrix=[X_element**i for i in range(len(X))]
    X_design_matrix_list.append(X_design_matrix)

X_=np.array(X_design_matrix_list)
X_T=X_.T

Y_=np.reshape(Y,[-1,1])

# print('計画行列',X_)
# print('計画行列(転置)',X_T)

#ハイパーパラメータ
def H_parameter(n): #n=0,1,2,3
    p=10**(-n*3) 
    return p
#リッジ回帰
def ParameterEstimation(Hyperparameter):
    w=np.linalg.inv((X_T@X_)+(Hyperparameter)*(np.eye(len(X_))))@X_T@Y
    return w
#回帰曲線
def Regressioncurve(w,x):
    y=w[0] + w[1] * x + w[2] * (x ** 2) + w[3] * (x ** 3) + w[4] * (x ** 4) + w[5] * (x ** 5) + w[6] * (x ** 6) + w[7] * (x ** 7) + w[8] * (x ** 8) + w[9] * (x ** 9) 
    return y


x_graph = np.linspace(np.min(X),np.max(X),1000)

#(1)α=1のとき
n=0
a1=H_parameter(n)
w1=ParameterEstimation(a1)
y1=Regressioncurve(w1,x_graph)
# #(2)α=10^(-3)のとき
n=1
a2=H_parameter(n)
w2=ParameterEstimation(a2)
y2=Regressioncurve(w2,x_graph)
# #(3)α=10^(-6)のとき
n=2
a3=H_parameter(n)
w3=ParameterEstimation(a3)
y3=Regressioncurve(w3,x_graph)
# #(4)α=10^(-9)のとき
n=3
a4=H_parameter(n)
w4=ParameterEstimation(a4)
y4=Regressioncurve(w4,x_graph)

# #グラフ描画
# Xmin, Xmax = 0, 1
# Ymin, Ymax = -1, 1
# delta=0.1
# plt.figure(figsize=(5,5))
# plt.grid(zorder=1)
# plt.scatter(X,Y,marker='o',color='b',zorder=2) 
# plt.xlabel('x') #x軸のラベル
# plt.ylabel('y') #y軸のラベル
# plt.xlim(Xmin-delta, Xmax+delta)
# plt.ylim(Ymin-delta, Ymax+delta)
# plt.plot(x_graph,y1,color='m',zorder=3)
# plt.plot(x_graph,y2,color='c',zorder=4)
# plt.plot(x_graph,y3,color='y',zorder=5)
# plt.plot(x_graph,y4,color='r',zorder=6)
# plt.show()

#2
def norm(w):
    norm=np.linalg.norm(w)
    return norm

#(1)α=1のとき
L2_1=norm(w1)
# #(2)α=10^(-3)のとき
L2_2=norm(w2)
# #(3)α=10^(-6)のとき
L2_3=norm(w3)
# #(4)α=10^(-9)のとき
L2_4=norm(w4)

#3
X_valid = np.array([ 0.05,  0.08,  0.12,  0.16,  0.28,  0.44,  0.47,  0.55,  0.63,  0.99])
Y_valid = np.array([ 0.35,  0.58,  0.68,  0.87,  0.83,  0.45,  0.01, -0.36, -0.83, -0.06])


#回帰曲線(L2正則化)
def Regressioncurve_norm(w,x,a):
    y=w[0] + w[1] * x + w[2] * (x ** 2) + w[3] * (x ** 3) + w[4] * (x ** 4) + w[5] * (x ** 5) + w[6] * (x ** 6) + w[7] * (x ** 7) + w[8] * (x ** 8) + w[9] * (x ** 9) + a * ((np.linalg.norm(w)) ** 2)
    return y
#MSR
def MSR(y_valid):
    diff=Y_valid-y_valid
    diff_2=np.power(diff,2)
    MSR=sum(diff_2)/len(X_valid)
    return MSR

#(1)α=1のとき
n=0
a1=H_parameter(n)
y1_valid=Regressioncurve_norm(w1,X_valid,a1)
MSR_1=MSR(y1_valid)
# #(2)α=10^(-3)のとき
n=1
a2=H_parameter(n)
y2_valid=Regressioncurve_norm(w2,X_valid,a2)
MSR_2=MSR(y2_valid)
# #(3)α=10^(-6)のとき
n=2
a3=H_parameter(n)
y3_valid=Regressioncurve_norm(w3,X_valid,a3)
MSR_3=MSR(y3_valid)
# #(4)α=10^(-9)のとき
n=3
a4=H_parameter(n)
y4_valid=Regressioncurve_norm(w4,X_valid,a4)
MSR_4=MSR(y4_valid)

#汎化性能判断
def Minimum(m1,m2,m3,m4):
    if m1<m2 and m1<m3 and m1<m4:
        print('α=10^(-9)のとき')
    elif m2<m1 and m2<m3 and m2<m4:
        print('α=10^(-6)のとき')
    elif m3<m1 and m3<m2 and m3<m4:
        print('α=10^(-3)のとき')
    else:
        print('α=1のとき')


Minimum(MSR_1,MSR_2,MSR_3,MSR_4)