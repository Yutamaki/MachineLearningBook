from colorsys import yiq_to_rgb
from re import A
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.animation
from IPython.display import HTML
import japanize_matplotlib

#確認問題

D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])

X = D[:,0]
Y = D[:,1]

X_bar=sum(X)/len(D)
Y_bar=sum(Y)/len(D)

#1
def linearregression(D):
    covXY=(1/len(D))*sum((X-X_bar)*(Y-Y_bar))
    varX=(1/len(D))*sum((X-X_bar)**2)
    a=covXY/varX
    b=Y_bar-a*X_bar
    return a,b
    
a=linearregression(D)[0]
b=linearregression(D)[1]

print('a=',a)
print('b=',b)


#2
Xmin, Xmax = 0, 10
Ymin, Ymax = 0, 10
x = np.arange(Xmin, Xmax, 0.1)  
y = a* x+b  

plt.figure(figsize=(5,5))
plt.grid(zorder=1)
plt.plot(x,y,c='red',zorder=2)
plt.scatter(X,Y,zorder=3) 
plt.xlim(Xmin, Xmax)
plt.ylim(Ymin, Ymax)
plt.show()


#3
def residualerror(D):
    residualerrorlist=[]
    for i in range(len(D)):
        residualerrorlist.append(D[i][1]-a*D[i][0]+b)
    return residualerrorlist
        

residualerrorlist=residualerror(D)

ε１=residualerrorlist[0]
ε２=residualerrorlist[1]
ε3=residualerrorlist[2]
ε4=residualerrorlist[3]

print('ε1=',ε１)
print('ε2=',ε2)
print('ε3=',ε3)
print('ε4=',ε4)

#4
#残差をZとすると，
Z=residualerrorlist
Z_bar=sum(Z)/len(Z)

covXZ=(1/len(Z))*sum((X-X_bar)*(Z-Z_bar))

print('説明変数と残差の共分散=',covXZ)

#5
covYZ=(1/len(Z))*sum((Y-Y_bar)*(Z-Z_bar))

print('目的変数の推定値と残差の共分散=',covYZ)

#6
varZ=(1/len(Z))*sum((Z-Z_bar)**2)
varY=(1/len(Y))*sum((Y-Y_bar)**2)

R=1-(varZ/varY)

print('決定係数R^2=',R)