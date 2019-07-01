#coding:utf-8
#author:cfx
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
主成分降维：将一组可能线性相关的变量，通过正交变换转换成一组线性不相关的变量
"""
"""
训练模型
求特征向量
求主成分系数
"""

"""from sklearn.decomposition import PCA
import pandas as pd
file='principal_component.xls'
data=pd.read_excel(file)

pca=PCA(3)
pca.fit(data)
pca.transform(data)
print pca.components_
print pca.explained_variance_ratio_"""
"""
KNN分类
"""
"""data2=make_blobs(n_samples=500,centers=5,random_state=8)
X2,y2=data2
import numpy as np
clf=KNeighborsClassifier()
clf.fit(X2,y2)
plt.scatter(X2[:,0],X2[:,1],c=y2,cmap=plt.cm.spring,edgecolors='k')
plt.show()"""
"""
KNN 线性回归
"""
"""from sklearn.datasets import make_regression
x,y=make_regression(n_features=1,n_informative=1,noise=50,random_state=8)
#plt.scatter(x,y,c='orange',edgecolors='k')
from sklearn.neighbors import KNeighborsRegressor
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(x,y)
import numpy as np
z=np.linspace(-3,3,200).reshape(-1,1)
plt.scatter(x,y,c='orange',edgecolors='k')
plt.plot(z,reg.predict(z),c='k',linewidth=3)
plt.title('KNN Regressor')
print reg.score(x,y)
plt.show()
"""

"""
Kmeans 聚类

1.在n个样本中随机选取K个点作为聚类中心
2.计算剩下的其他点到这几个聚类中心的距离，最近的归为那个类下
3.重新选取k个点作为聚类中心，重复2步骤
4.直至聚类不再发生改变

"""
"""
from sklearn.cluster import KMeans
file='consumption_data.xls'
data=pd.read_excel(file)
print data.head()
k=4
iteration=500##迭代次数
data_zs=1.0*(data-data.mean())/data.std()##数据标准化
#print data_zs.head()
model=KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)
model.fit(data_zs)
r1=pd.Series(model.labels_).value_counts() ##统计分类的情况
r2=pd.DataFrame(model.cluster_centers_)##求聚类中心
#r=pd.concat([r2,r1],axis=1)
#print r
r=pd.concat([data,pd.Series(model.labels_,index=data.index)],axis=1)
#print r
from sklearn.manifold import TSNE
tsne=TSNE()
tsne.fit_transform(data_zs)
tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index)

d=tsne[r[0]==0]
plt.plot(d[0],d[1],'r.')
d=tsne[r[0]==1]
plt.plot(d[0],d[1],'y.')
d=tsne[r[0]==2]
plt.plot(d[0],d[1],'go')
d=tsne[r[0]==3]
plt.plot(d[0],d[1],'b*')
plt.show()"""

"""
关联规则

"""
def getDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
def createC1(dataSet):
    C1=[]
    for data in dataSet:
        #print data
        for item in data:
            #print item
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    print C1
    return map(frozenset, C1)
#print createC1(getDataSet())
def scanData(D,Ck,minSuport):
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(D))
    retlist=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems
        if support>=minSuport:
            retlist.insert(0,key)
        supportData[key]=support
    return retlist,supportData
def aprioriGen(Lk,k):
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList
def apriori(dataSet, minSupport = 0.5):
    C1=createC1(dataSet)
    D=map(set,dataSet)
    L1,supportData=scanData(D,C1,minSupport)
    L=[L1]
    k=2
    while(len(L[k-2])>0):
        Ck=aprioriGen(L[k-2],k)
        Lk,supK=scanData(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData
dataSet=getDataSet()
C1=createC1(dataSet)

print '所有的候选集为：',C1
D=map(set,dataSet)
print "数据集为：",D

L1,supprtData=scanData(D,C1,0.5)
print '符合最小支持度的频繁1项集L1:',L1

L,superData=apriori(dataSet)
print "所有符合最小支持度的项集L：\n",L
print "频繁3项集：\n",aprioriGen(L[0],3)

L,superData=apriori(dataSet,0.7)
print "所有符合最小支持度的项集为:",L
print "频繁2项集",aprioriGen(L[0],2)

