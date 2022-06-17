from re import I
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
 
#確認問題

D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
x = D[:,0]
y = D[:,1]

X1=np.hstack([np.ones(len(D)).reshape(-1,1),D[:,:-1]])
Y=y
t=y

print('Xは',X1)
print('Yは',Y)

#1
#この場合Yが目標値
#X^T*Xの計算
XTX=X1.T@X1
print('XTXは',XTX)

XTX_INV=np.linalg.inv(XTX)
print('XTXINVは',XTX_INV)

XTY=X1.T@Y
print('XTYは',XTY)

w=XTX_INV@XTY

print('wは',w)

w1_1=np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(Y)

print('w1_1は',w1_1)

w1_2=np.linalg.solve(X1.T.dot(X1),X1.T.dot(Y))

print('w1_2は',w1_2)

print('ω1=',w[1])
print('ω0=',w[0])


#2
X2=np.hstack([np.ones(len(D)).reshape(-1,1),D[:,:-1],(D[:,:-1])**2])
Y=y
t=y


w2_1=np.linalg.inv(X2.T@X2)@X2.T@Y
print('w2_1は',w2_1)
w2_2=np.linalg.solve(X2.T@X2,X2.T@Y)
print('w2_2は',w2_2)

ω0,ω1,ω2=w2_1[0],w2_1[1],w2_1[2]

print('ω2=',ω2)
print('ω1=',ω1)
print('ω0=',ω0)


#3
# Xmin, Xmax = 0, 10
# Ymin, Ymax = 0, 10
# plt.figure(figsize=(5,5))
# plt.grid(zorder=1)
# plt.scatter(x,y,zorder=2) 
# x = np.arange(0, 11,0.01)
# y =ω2* x ** 2 + ω1 * x + ω0
# plt.xlabel('x') #x軸のラベル
# plt.ylabel('y') #y軸のラベル
# plt.xlim(Xmin, Xmax)
# plt.ylim(Ymin, Ymax)
# plt.plot(x,y, linestyle='-', color='blue', marker='',zorder=3) 
# plt.show()


#4
D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
x = D[:,0]
y = D[:,1]

X2=np.hstack([np.ones(len(D)).reshape(-1,1),D[:,:-1],(D[:,:-1])**2])
Y=y
t=y

varε=((y-(X2@w2_1))**2).sum()
varY=((y-y.mean())**2).sum()

R=1-(varε/varY)

print('決定係数R^2=',R)

#5
X3=np.hstack([np.ones(len(D)).reshape(-1,1),D[:,:-1],(D[:,:-1])**2,(D[:,:-1])**3])
Y=y
t=y

w3_1=np.linalg.inv(X3.T@X3)@X3.T@Y
print('w3_1は',w3_1)
w3_2=np.linalg.solve(X3.T@X3,X3.T@Y)
print('w3_2は',w3_2)

ω0,ω1,ω2,ω3=w3_1[0],w3_1[1],w3_1[2],w3_1[3]

print('ω3=',ω3)
print('ω2=',ω2)
print('ω1=',ω1)
print('ω0=',ω0)

#6
Xmin, Xmax = 0, 10
Ymin, Ymax = 0, 10
plt.figure(figsize=(5,5))
plt.grid(zorder=1)
plt.scatter(x,y,zorder=2) 
x = np.arange(0, 11,0.01)
y =ω3* x ** 3 + ω2* x ** 2 + ω1 * x + ω0
plt.xlabel('x') #x軸のラベル
plt.ylabel('y') #y軸のラベル
plt.xlim(Xmin, Xmax)
plt.ylim(Ymin, Ymax)
plt.plot(x,y, linestyle='-', color='blue', marker='',zorder=3) 
plt.show()

#7
D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
x = D[:,0]
y = D[:,1]

X3=np.hstack([np.ones(len(D)).reshape(-1,1),D[:,:-1],(D[:,:-1])**2,(D[:,:-1])**3])
Y=y
t=y

varε=((y-(X3@w3_1))**2).sum()
varY=((y-y.mean())**2).sum()

R=1-(varε/varY)

print('決定係数R^2=',R)