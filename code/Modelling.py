import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import optuna

data = pd.read_csv("DES 58 3.csv")  # 替换为你的数据文件路径
label_column = data.columns[-1]  # 假设最后一列是标签列
X = data.drop(columns=[label_column])  # 特征集
y = data[label_column]  # 标签列

# 分别处理类别0和类别1
X_0 = X[y == 0]
y_0 = y[y == 0]
X_1 = X[y == 1]
y_1 = y[y == 1]

# 分配类别0的数据
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, train_size=30, test_size=8, random_state=0)
# 分配类别1的数据
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, train_size=16, test_size=4, random_state=0)

# 合并训练集和测试集
X_train = pd.concat([X_train_0, X_train_1])
y_train = pd.concat([y_train_0, y_train_1])
X_test = pd.concat([X_test_0, X_test_1])
y_test = pd.concat([y_test_0, y_test_1])
# 定义 Optuna 的 objective 函数（ADA）
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 1000)  # AdaBoost 的弱学习器数量
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 10, log=True)  # 学习率

    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm="SAMME",  # 显式指定使用 SAMME 算法
        random_state=0
    )
    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model,  X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)
# 定义 Optuna 的 objective 函数（LGBM）
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 4, 14)  # 树的最大深度
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1, log=True)  # 学习率
    n_estimators = trial.suggest_int("n_estimators", 50, 1000)  # 迭代次数
    num_leaves = trial.suggest_int("num_leaves", 10, 100)  # 每棵树的叶子数
    lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0)  # L1 正则化
    lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0)  # L2 正则化
    min_child_samples = trial.suggest_int("min_child_samples", 4, 100)  # 新增超参数

    model = LGBMClassifier(
         max_depth=max_depth,
         learning_rate=learning_rate,
         n_estimators=n_estimators,
         num_leaves=num_leaves,
         lambda_l1=lambda_l1,
         lambda_l2=lambda_l2,
         min_child_samples=min_child_samples,
         random_state=0,
         verbose=-1  # 禁用日志输出
    )
    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)
# 定义 Optuna 的 objective 函数(LR)
def objective(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    C = trial.suggest_float("C", 1e-10, 1e+10,log=True)  # 浮点型，对数分布
    solver = "liblinear" if penalty == "l1" else "lbfgs"  # 根据 penalty 选择合适的 solver

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        random_state=0,
        max_iter=1000  # 增加最大迭代次数以确保收敛
    )
    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)
# 定义 Optuna 的 objective 函数(XGB)
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1200, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=0,
        use_label_encoder=False,  # 避免警告
        eval_metric="logloss"  # 避免警告
    )
    # 交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)
# 定义 Optuna 的 objective 函数(RF)
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, 1)  # 整数型，(参数名称，下界，上界，步长)
    max_depth = trial.suggest_int("max_depth", 5, 20, 1)
    max_features = trial.suggest_int("max_features", 5, 30, 1)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0, 5, log=False)  # 浮点型

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_impurity_decrease=min_impurity_decrease,
        random_state=0,
        verbose=False,
        n_jobs=8
    )
    # 交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)
# 定义 Optuna 的 objective 函数(CAT)
def objective(trial):
    depth = trial.suggest_int("depth", 3, 10)  # 树的深度
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1, log=True)  # 学习率
    n_estimators = trial.suggest_int("n_estimators", 50, 500)  # 迭代次数
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 10)  # L2 正则化

    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        l2_leaf_reg=l2_leaf_reg,
        random_state=0,
        verbose=0  # 禁用日志输出
    )
    # 交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)
# 定义 Optuna 的 objective 函数（DT）
def objective(trial):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 2, 40)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=0
    )
    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model,  X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)

# 使用 Optuna 进行超参数优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  # 运行100次试验

# 打印最佳超参数
best_params = study.best_params
print("最佳超参数：", best_params)

# 使用最佳超参数重新训练模型 ADA
best_model = AdaBoostClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    algorithm="SAMME",  # 显式指定使用 SAMME 算法
    random_state=0
)
# 使用最佳超参数重新训练模型 LGBM
best_model = LGBMClassifier(
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    num_leaves=best_params['num_leaves'],
    lambda_l1=best_params['lambda_l1'],
    lambda_l2=best_params['lambda_l2'],
    random_state=0,
    verbose=-1  # 禁用日志输出
)
# 使用最佳超参数重新训练模型 LR
solver = "liblinear" if best_params['penalty'] == "l1" else "lbfgs"
best_model = LogisticRegression(
    penalty=best_params['penalty'],
    C=best_params['C'],
    solver=solver,
    random_state=0,
    max_iter=1000  # 增加最大迭代次数以确保收敛
)
# 使用最佳超参数重新训练模型 XGB
best_model = XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    random_state=0,
    use_label_encoder=False,  # 避免警告
    eval_metric="logloss"  # 避免警告
)
# 使用最佳超参数重新训练模型 RF
best_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_impurity_decrease=best_params['min_impurity_decrease'],
    random_state=0,
    verbose=False,
    n_jobs=8
)
# 使用最佳超参数重新训练模型 CAT
best_model = CatBoostClassifier(
    depth=best_params['depth'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    random_state=0,
    verbose=0  # 禁用日志输出
)
# 使用最佳超参数重新训练模型 DT
best_model = DecisionTreeClassifier(
    criterion=best_params['criterion'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=0
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# 定义RFE对象
rfecv = RFECV(estimator=best_model, min_features_to_select=5, step=1, cv=skf, scoring='accuracy')  # 最小特征数量为5，每次消除1个特征
rfecv.fit(X_train, y_train)
selected_features = X_train.columns[rfecv.support_]
print("被选择的特征：", selected_features)
X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)

# 评估模型
accuracy_scores = cross_val_score(best_model, X_train_rfecv, y_train, cv=skf, scoring='accuracy')
recall_scores = cross_val_score(best_model, X_train_rfecv, y_train, cv=skf, scoring='recall_weighted')
roc_auc_scores = cross_val_score(best_model, X_train_rfecv, y_train, cv=skf, scoring='roc_auc_ovr_weighted')
f1_scores = cross_val_score(best_model, X_train_rfecv, y_train, cv=skf, scoring='f1_weighted')  # 加入 F1 分数
# 打印评估结果
print(f"平均准确率：{np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"平均召回率：{np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"平均AUC：{np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}")
print(f"平均F1分数：{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")  # 输出 F1 分数

# 测试集评估
best_model.fit(X_train_rfecv, y_train)  # 在训练集上重新训练模型
y_pred = best_model.predict(X_test_rfecv)  # 对测试集进行预测

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

# 计算测试集的评估指标
test_accuracy = accuracy_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred, average='weighted')
test_roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
test_f1 = f1_score(y_test, y_pred, average='weighted')

# 打印测试集的评估结果
print(f"测试集准确率：{test_accuracy:.4f}")
print(f"测试集召回率：{test_recall:.4f}")
print(f"测试集AUC：{test_roc_auc:.4f}")
print(f"测试集F1分数：{test_f1:.4f}")
# 保存选择的特征
output_file_path = 'selected_features().csv'
pd.DataFrame(selected_features, columns=['Feature']).to_csv(output_file_path, index=False)
print(f"选择的特征已保存到 {output_file_path}")

# 获取特征重要性分数
best_model.fit(X_train_rfecv, y)  # 在全部数据上训练模型以获取特征重要性
feature_importances = best_model.feature_importances_
feature_names = selected_features

# 创建一个DataFrame来保存特征重要性分数
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.round(feature_importances, 4)})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 保存特征重要性分数到CSV文件
feature_importance_output_path = 'feature_importance().csv'
feature_importance_df.to_csv(feature_importance_output_path, index=False)
print(f"特征重要性分数已保存到 {feature_importance_output_path}")
# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.title('Feature Importance', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# 保存图片
plt.savefig('feature_importance（）.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()