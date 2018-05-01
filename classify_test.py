# -*- coding: utf-8 -*-
"""
分类方法的筛选。剔除分类结果较差的方法
"""

import pandas as pd
pd_data=pd.read_excel('Data4Roger.xlsx')

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#h = .02  # step size in the mesh
#"Linear SVM"
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(3),  #k-近邻
    SVC(kernel="linear", C=0.025), #线性核SVM
    SVC(gamma=2, C=1),    #径向基SVM
    DecisionTreeClassifier(max_depth=5), #决策树
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  #随机森林
    AdaBoostClassifier(),   #AdaBoost
    GaussianNB(),  #贝叶斯方法 
    LinearDiscriminantAnalysis(),  #线性判别分析
    QuadraticDiscriminantAnalysis()] #二次判别分析

sum_1=sum_2=sum_3=sum_4=sum_5=sum_6=sum_7=sum_8=sum_9=0.0
for i in range(0,100):
    #print(i)
    pd_data=pd_data.sample(frac=1) #打乱顺序
    pd_y=pd_data['group']
    pd_x=pd_data.drop(['subjectID','subjnumber','group'],axis=1)
    x=pd_x.values
    y=list(pd_y)
    for name, clf in zip(names, classifiers):
        #print(name)
        '''
        clf.fit(x,y)
        expected=y
        predicted=clf.predict(x)
        acc=clf.score(x, y)
        '''
        n=len(y)/4*3
        clf.fit(x[:n],y[:n])
        #expected=y[n:]
        #predicted=clf.predict(x[n:])
        acc=clf.score(x[n:], y[n:]) 
        if(name=="Nearest Neighbors"):
            sum_1=sum_1+acc
        if(name=="Linear SVM"):
            sum_2=sum_2+acc
        if(name=="RBF SVM"):
            sum_3=sum_3+acc
        if(name=="Decision Tree"):
            sum_4=sum_4+acc
        if(name=="Random Forest"):
            sum_5=sum_5+acc
        if(name=="AdaBoost"):
            sum_6=sum_6+acc
        if(name=="Naive Bayes"):
            sum_7=sum_7+acc
        if(name=="Linear Discriminant Analysis"):
            sum_8=sum_8+acc
        if(name=="Quadratic Discriminant Analysis"):
            sum_9=sum_9=acc

average_acc_1=sum_1/(i+1)
average_acc_2=sum_2/(i+1)
average_acc_3=sum_3/(i+1)
average_acc_4=sum_4/(i+1)
average_acc_5=sum_5/(i+1)
average_acc_6=sum_6/(i+1)
average_acc_7=sum_7/(i+1)
average_acc_8=sum_8/(i+1)
average_acc_9=sum_9/(i+1)
print(u'预测准确率')
print("Nearest Neighbors:%f" %(average_acc_1))
print("Linear SVM:%f" %(average_acc_2))
print("RBF SVM:%f" %(average_acc_3))
print("Decision Tree:%f" %(average_acc_4))
print("Random Forest:%f" %(average_acc_5))
print("AdaBoost:%f" %(average_acc_6))
print("Naive Bayes:%f" %(average_acc_7))
print("Linear Discriminant Analysis:%f" %(average_acc_8))
print("Quadratic Discriminant Analysis:%f" %(average_acc_9))

    
    
