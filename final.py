# -*- coding: utf-8 -*-
"""
剔除k-近邻、径向基SVM、二次判别分析，利用线性SVM、决策树、随机森林、AdaBoost、贝叶斯分类、线性鉴别分析进行组合预测
"""

import pandas as pd
pd_data=pd.read_excel('Data4Roger.xlsx') #读入数据

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

#h = .02  # step size in the mesh
#"Linear SVM"
names = ["Linear SVM","Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis"]
classifiers = [
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis()]

sum_1=sum_2=sum_3=sum_4=sum_5=sum_6=sum_7=sum_8=sum_9=0.0
sum=0.0
for num in range(0,100):
    #print(num)
    predict=[]
    #print(i)
    pd_data=pd_data.sample(frac=1) #随机打乱顺序
    pd_y=pd_data['group'] 
    pd_x=pd_data.drop(['subjectID','subjnumber','group'],axis=1)
    x=pd_x.values #分类指标
    y=list(pd_y) #类别
    n=len(y)/4*3 
    list_a=[0]
    list_a=list_a*(len(y)-n)
    a=pd.DataFrame(list_a)
    a.columns=['a']
    
    for name, clf in zip(names, classifiers):
        #print(name)
        '''
        clf.fit(x,y)
        expected=y
        predicted=clf.predict(x)
        acc=clf.score(x, y)
        '''
        #n=len(y)/4*3
        clf.fit(x[:n],y[:n]) #
        expected=y[n:]
        predicted=clf.predict(x[n:])
        #acc=clf.score(x[n:], y[n:])
        b=pd.DataFrame(list(predicted)) 
        b.columns=['b']
        a=pd.concat([a,b],axis=1) #连接两个DataFrame
        df_predict=a.drop(['a'],axis=1) #每种方法的预测结果组成的DataFrame
    #找到预测结果中出现次数最多的类别    
    for i in range(0,len(df_predict)):
        predict_user_y=list(df_predict.loc[i])
        b = {}
        for d in set(predict_user_y):    #去重复的值，set
            b[predict_user_y.count(d)] = d   #去重后做计数，把数量和值写到字典b
        for e in reversed(sorted(b.keys())[-1:]): 
            #print e,':',b[e]   #排序列表键值并取后3个（数量最大的3个），翻转后打印出数量与值
            predict.append(b[e])
            #predicted_label.append(b)   
    acc_model=accuracy_score(expected,predict)
    sum=sum+acc_model

average_acc=sum/(num+1)
print(u"平均预测准确率：%f\n" %(average_acc))



    
        
    
        
        