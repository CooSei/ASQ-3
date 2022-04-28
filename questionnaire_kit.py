import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
from imblearn.over_sampling import SMOTE
from dtreeplot import model_plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score

def efficiency(y_test, y_pred):
    '''
    机器学习预测评估
    :param y_test: 真实值
    :param y_pred: 预测值
    :return: 灵敏度、特异度、精确率、正确率、kappa
    '''
    _matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    recall = _matrix[0][0]/_matrix[0].sum()
    specificity = _matrix[1][1]/_matrix[1].sum()
    precision = _matrix[0][0]/(_matrix[0][0]+_matrix[1][0])
    accuracy = (_matrix[0][0]+_matrix[1][1])/_matrix.sum()
    kappa = cohen_kappa_score(y_test,y_pred)
    return round(recall, 4), round(specificity, 4), round(precision, 4), round(accuracy, 4), round(kappa, 4)


def drow_ROC(X_test, y_test, clf, rf):
    '''
    绘制ROC并展示保存
    :param X_test: 测试集X
    :param y_test: 测试集y
    :param clf: 决策树
    :param rf: 随机森林
    :return: None
    '''
    plt.rc('font',family="Times New Roman")
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(labelsize=12)

    fpr_tree, tpr_tree, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    tree_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    fpr_rf, tpr_rf, thresholds1 = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    plt.xlabel('1-Specificity', size=14)
    plt.ylabel('Sensitivity', size=14)
    plt.plot([0, 1], [0, 1],
             'k--',
             linewidth=1.5,
             label='Line of Reference',
             color=[7/255, 7/255, 7/255]
             )
    plt.plot(fpr_tree, tpr_tree,
             linewidth=1.5,
             label='CART',
             color=[56/255, 89/255, 173/255]
             )
    plt.plot(fpr_rf, tpr_rf,
             linewidth=1.5,
             label='Random Forest',
             color=[210/255, 32/255, 9/255]
             )
    plt.text(0.04, 0.95,
             "AUC = "+str(round(rf_auc, 2)),
             color=[210/255, 32/255, 39/255],
             size=14
             )
    plt.text(0.3, 0.7,
             "AUC = "+str(round(tree_auc, 2)),
             color=[56/255, 89/255, 173/255],
             size=14
             )
    plt.legend(loc='lower right')
    # plt.grid()
    plt.show()
    plt.savefig('roc.png', dpi=300)


def search_best_clf_parameter(X_train, y_train, *para):
    '''
    寻找决策树最优参数
    :param para: 元组形式输入深度范围，最小分割范围
    :return: 树深、最小分割点
    '''
    p1, p2, p3, p4 = para
    depth = np.arange(p1, p2)
    min_split = np.arange(p3, p4)
    clf = DecisionTreeClassifier(criterion='gini', random_state=40)
    params = {
        "max_depth": depth,
        'min_samples_split': min_split,
    }
    result = GridSearchCV(
        clf,
        param_grid=params,
        cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=40),
    )
    result.fit(X_train, y_train)
    return result.best_params_['max_depth'], result.best_params_['min_samples_split']


def search_random_forest_para(X_train1, y_train1,**paras):
    '''
    随机森林调参
    :param X_train:
    :param y_train:
    :param paras: 字典形式输入参数范围
    :return: 树数量，树深，特征数
    '''
    tree_number_range = paras['n']
    tree_depth = paras['depth']
    tree_features = paras['features']
    score_list = []
    for i in tree_number_range:
        rf1 = RandomForestClassifier(n_estimators=i, bootstrap=True, random_state=40)
        score = cross_val_score(rf1,
                                X_train1,
                                y_train1,
                                cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=40)).mean()
        score_list.append(score)
    score_max = max(score_list)
    rf1.set_params(n_estimators=tree_number_range[score_list.index(score_max)])
    score_list = []
    for i in tree_depth:
        rf1.set_params(max_depth=i)
        score = cross_val_score(rf1,
                                X_train1,
                                y_train1,
                                cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=40)).mean()
        score_list.append(score)
    score_max = max(score_list)
    rf1.set_params(max_depth=tree_depth[score_list.index(score_max)])
    score_list = []
    for i in tree_features:
        rf1.set_params(max_features=i)
        score = cross_val_score(rf1,
                                X_train1,
                                y_train1,
                                cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=40)).mean()
        score_list.append(score)
    score_max = max(score_list)
    rf1.set_params(max_features=tree_features[score_list.index(score_max)])
    # para_str = re.search('\((.*)\)', rf.__str__()).group(1)
    rf_paras = rf1.get_params()
    return rf_paras['n_estimators'], rf_paras['max_depth'], rf_paras['max_features']


def rf_feature_stepwise(sub_data,feature_numbers):
    x, y = sub_data.iloc[:, 1:], sub_data.iloc[:, 0]
    X_train_, X_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.3, random_state=40)
    X_train_, y_train_ = SMOTE(random_state=50).fit_resample(X_train_, y_train_)
    n, d, f = search_random_forest_para(X_train_, y_train_,
                                        n=range(1, 500, 10),
                                        depth=range(1, 20),
                                        features=range(1, feature_numbers)
                                        )
    sub_rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    max_features=f,
                                    random_state=40
                                    )
    sub_rf.fit(X_train_, y_train_)
    return (n, d, f) + efficiency(y_test_, sub_rf.predict(X_test_))


if __name__ == '__main__':
    df = pd.read_excel('整合数据.xls')
    learn_data = df[['y', 'Sex', 'Q2', 'Q4', 'Q5', 'Q6', 'Q30', 'Q21', 'Q22', 'Q1', 'Q3', 'Q26', 'Q13', 'Q19', 'Q27']]
    x_1, y_1 = learn_data.iloc[:, 1:], learn_data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(x_1, y_1, test_size=0.3, random_state=40)
    X_train, y_train = SMOTE(random_state=50).fit_resample(X_train, y_train)
    clf_depth, clf_min_split = search_best_clf_parameter(X_train, y_train, 1, 20, 2, 50)
    clf = DecisionTreeClassifier(max_depth=clf_depth,
                                 min_samples_split=clf_min_split,
                                 random_state=40
                                 )
    clf.fit(X_train, y_train)
    model_plot(clf, X_train.columns, labels=y_train, height=500, show_notebook=False)
    n, d, f = search_random_forest_para(X_train, y_train,
                                        n=range(1, 500, 10),
                                        depth=range(1, 20),
                                        features=range(3, 15)
                                        )
    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, random_state=40)
    rf.fit(X_train, y_train)
    tree_r, tree_s, tree_p, tree_a, tree_k = efficiency(y_test, clf.predict(X_test))
    rf_r, rf_s, rf_p, rf_a, rf_k = efficiency(y_test, rf.predict(X_test))
    print('模型 %5s灵敏度 %3s特异度 %3s精确率 %3s正确率 %3skappa'%('', ' ', ' ', ' ', ' '))
    print('决策树 %3s%.4f%3s%.4f%4s%.4f%3s%.4f%3s%.4f' % (' ', tree_r, ' ', tree_s, ' ', tree_p, ' ', tree_a, ' ', tree_k))
    print('随机森林 %1s%.4f%3s%.4f%4s%.4f%3s%.4f%3s%.4f' % (' ', rf_r, ' ', rf_s, ' ', rf_p, ' ', rf_a, ' ', rf_k))
    # 开始特征重要性排序
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)[1]
    x = abs(shap_values).sum(axis=0)
    feature_importance_df = pd.DataFrame(index=X_train.columns, data=x, columns=['values'])
    feature_importance_df.sort_values(by='values', ascending=False, inplace=True)
    feature_sort_list = feature_importance_df.index
    evalute_df = pd.DataFrame(columns=['n', 'depth', 'features', '灵敏度', '特异度', '精确率', '准确率', 'kappa'])
    new_feature_list = ['y']
    for i in feature_sort_list:
        new_feature_list.append(i)
        result = rf_feature_stepwise(learn_data[new_feature_list], len(new_feature_list))
        evalute_df.loc[len(new_feature_list)-1] = result
    print(evalute_df)




