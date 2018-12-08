from sklearn.svm import SVC
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def iris_type(s):
    it = {b"b": 0,b"f": 1,b"s": 2,b"y": 3}
    return it[s]

path = './fisher_data1.txt'  # 数据文件路径
save_name = 'fisher_data1.xml'
file_result = 'fisher_data1.txt'

data = np.loadtxt(path, dtype=float, delimiter=',', converters={-1: iris_type})
print(data.shape)
lenth = len(data[1,:])
width = len(data[:,-1])
x, y = np.split(data, ((lenth-1),), axis=1)#最后一列为标签
y = (y.ravel()).astype(np.int)
print("特征向量长度：",x.shape)
print("标签值：",y)

f = open(file_result, 'w')
f.write(file_result+"\n")
f.write("特征向量长度：" + str(x.shape)+"\n")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.8)
print('x_train.shape:{0};\nx_test.shape:{1};\ny_train.shape:{2};\ny_test.shape:{3}'.format(x_train.shape,x_test.shape,y_train.shape,y_test.shape))

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]#参数集合

scores = ['precision', 'recall']#评分策略


for score in scores:
    print("# Tuning hyper-parameters for %s\n" % score)
    f.write("# Tuning hyper-parameters for "+ score+'\n')
    #构造这个GridSearch的分类器,5-fold 
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_weighted' % score) 
    #只在训练集上面做k-fold,然后返回最优的模型参数 
    clf.fit(x_train, y_train) 
    ##保存网络模型
    joblib.dump(clf,save_name)
    print("训练精度：",clf.score(x_train, y_train))
    print("测试精度：",clf.score(x_test, y_test))
    f.write("训练精度："+str(clf.score(x_train, y_train))+'\n')
    f.write("测试精度："+str(clf.score(x_test, y_test))+'\n')
    print("Best parameters set found on development set:\n") 
    f.write("Best parameters set found on development set:\n")
    #输出最优的模型参数 
    print(clf.best_params_) 
    f.write(str(clf.best_params_)+'\n')
    print() 
    print("Grid scores on development set:\n") 
    f.write("Grid scores on development set:\n")

    #clf.cv_results_是一个字典类型的变量 
    #第一类是时间， 第二类是参数， 第三类是测试分数，其中又分为每次交叉验证的参数和统计的参数
    #第四类是训练分数，其中也分为每次交叉验证的参数和统计的参数。
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    #各种参数使用交叉验证的时候的均值和方差
    for mean, std, params in zip(means, stds, clf.cv_results_['params']): 
        print("mean：%0.3f，std： (+/-%0.03f) for %r\n" % (mean, std * 2, params))
        f.write("mean：" + str(('%0.3f'%mean))+ ",std：(+-)" + str(('%0.3f'%(std * 2))) +str(params)+'\n')

    print("Detailed classification report:\n") 
    f.write("Detailed classification report:\n")
    print("The model is trained on the full development set.") 
    f.write("The model is trained on the full development set."+'\n')
    print("The scores are computed on the full evaluation set.") 
    f.write("The scores are computed on the full evaluation set."+'\n')
    print() 
    #在测试集上测试最优的模型的泛化能力. 
    y_true, y_pred = y_test, clf.predict(x_test) 
    print(classification_report(y_true, y_pred)) 
    f.write(str(classification_report(y_true, y_pred))+'\n')














