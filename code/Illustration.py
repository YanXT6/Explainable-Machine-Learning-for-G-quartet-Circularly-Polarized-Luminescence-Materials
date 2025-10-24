import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RepeatedKFold, cross_val_score, StratifiedKFold, train_test_split
import optuna
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
from scipy.interpolate import make_interp_spline
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Fig. 2e-f
# 假设数据集是一个CSV文件，第一列是标签，其余列是特征
data = pd.read_csv('DES 58 3.csv')

# 分离标签和特征
labels = data.iloc[:, -1].values  # 提取最后一列作为标签
features = data.iloc[:, 1:].values

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 进行PCA降维至3维
pca = PCA(n_components=3,random_state=0)
pca_result = pca.fit_transform(features_scaled)

# 输出PCA的解释方差比例
print("PCA的解释方差比例:", pca.explained_variance_ratio_)
print(pca.explained_variance_)

PCA_9 = pd.DataFrame(pca_result)
PCA_9.columns = ["PCA1", "PCA2", "PCA3"]
PCA_9['label'] = labels
# 将PCA结果保存到新的CSV文件
PCA_9.to_csv('.csv', index=False, sep=',')  # 保存到CSV文件

# 分离两组数据
group1 = PCA_9[PCA_9['label'] == 0]  # 假设标签0为第一组
group2 = PCA_9[PCA_9['label'] == 1]  # 假设标签1为第二组

# 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制第一组数据
ax.scatter(group1['PCA1'], group1['PCA2'], group1['PCA3'], alpha=1, c='#0064d2', label='NO CPL', marker='x',s=50)
# 绘制第二组数据
ax.scatter(group2['PCA1'], group2['PCA2'], group2['PCA3'], alpha=0.8, c='#e53238', label='CPL', marker='o',s=50)
# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'
# 设置图例和标签
ax.set_xlabel('PC1',fontsize=14, rotation=-20)
ax.set_ylabel('PC2',fontsize=14, rotation=55)
ax.set_zlabel('PC3',fontsize=14, rotation=90)
plt.title('Features = 54', fontsize=20, fontweight='bold')
# 设置坐标轴刻度
plt.tick_params(labelsize=12)
ax.legend()
# 保存图片
plt.savefig('.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()

#Fig. 3a
sheet_name = "Sheet1"
# 读取数据
df = pd.read_excel("All model.xlsx", sheet_name=sheet_name)

# 2. 数据预处理
df['Features'] = df['Features'].apply(lambda x: 'RFE' if 'RFE' in str(x) else str(x))

# 设置特征类别顺序
feature_order = ['1355', '210', '54', 'RFE']

# 设置模型顺序（按字母顺序）
model_order = df['Models'].unique()

# 3. 创建数据透视表
# Accuracy透视表（模型 vs 特征）
acc_pivot = df.pivot(index="Features", columns="Models", values="Accuracy")
acc_pivot = acc_pivot.reindex(index=feature_order, columns=model_order)

# AUC透视表（特征 vs 模型）
auc_pivot = df.pivot(index="Features", columns="Models", values="AUC")
auc_pivot = auc_pivot.reindex(index=feature_order, columns=model_order)

# 4. 自定义颜色映射
# 为Accuracy创建蓝色渐变
acc_cmap = LinearSegmentedColormap.from_list(
    "acc_cmap", ["#edf5fd", "#86bff1", "#1e88e5"]
)

# 为AUC创建红色渐变
auc_cmap = LinearSegmentedColormap.from_list(
    "auc_cmap", ["#ffebf1", "#ff7ca4", "#ff0D57"]
)

# 5. 创建双热力图
# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(20, 10))

# 创建子图网格
gridspec = {'width_ratios': [1,1], 'wspace': 0.1}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw=gridspec)

# Accuracy热力图（模型 vs 特征）
sns.heatmap(
    acc_pivot,
    annot=True,
    fmt=".3f",
    cmap=acc_cmap,
    linewidths=1,
    linecolor='white',
    ax=ax1,
    cbar_kws={'shrink': 1}, #'label': 'Accuracy'},
    vmin=0.80,
    vmax=0.93,
    annot_kws={'size': 14, 'color': 'black'}
)
ax1.set_title('Accuracy', fontsize=24, pad=20, fontweight='bold')
ax1.set_xlabel('Models', fontsize=24, fontweight='bold')
ax1.set_ylabel('Features', fontsize=24, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=14)

# AUC热力图（特征 vs 模型）
sns.heatmap(
    auc_pivot,
    annot=True,
    fmt=".3f",
    cmap=auc_cmap,
    linewidths=1,
    linecolor='white',
    ax=ax2,
    cbar_kws={'shrink': 1}, #'label': 'AUC'},
    vmin=0.80,
    vmax=0.95,
    annot_kws={'size': 14, 'color': 'black'}
)
ax2.set_title('AUC', fontsize=24, pad=20, fontweight='bold')
ax2.set_xlabel('Models', fontsize=24, fontweight='bold')
ax2.set_ylabel('Features', fontsize=24, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=14)

plt.savefig('model_performance_heatmap.png', dpi=600, bbox_inches='tight')
plt.show()

# Fig. 3b
models = ['CAT', 'XGB', 'DT', 'ADA', 'LGBM', 'LR', 'RF']
features = [54, 54, 5, 9, 5, 10, 54]
accuracy = [0.9167, 0.8959, 0.875, 0.875, 0.85, 0.8333, 0.8542]
recall = [0.9167, 0.8959, 0.875, 0.875, 0.85, 0.8333, 0.8542]
auc = [0.875, 0.8594, 0.8594, 0.8438, 0.825, 0.8125, 0.7813]
f1 = [0.9132, 0.8932, 0.8737, 0.8738, 0.8497, 0.8337, 0.8394]
# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 创建雷达图
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, polar=True)

# 角度设置
categories = ['Accuracy', 'Recall', 'AUC', 'F1 Score']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 科研级美化设置
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.titlesize': 24
})

# 特殊突出显示0.85和0.9的圆圈
for radius, color in [(0.85, '#C3BEE7'), (0.9, '#B5DFC9')]:
    # 创建圆形路径
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    circle_x = radius * np.cos(circle_angles)
    circle_y = radius * np.sin(circle_angles)

    # 绘制特殊圆圈
    ax.plot(circle_angles, [radius] * 100,
            color=color,
            linewidth=1.5,
            linestyle='--',
            alpha=1)

# 绘制每个模型
colors = ["#1e88e5", "#52a3eb", "#86bff1", "#ffb4cb", "#ff7ca4", "#ff457e", "#ff0D57"]  # SHAP渐变

for i, model in enumerate(models):
    values = [accuracy[i], recall[i], auc[i], f1[i]]
    values += values[:1]  # 闭合多边形

    # 绘制主线
    ax.plot(angles, values, 'o-', color=colors[i], linewidth=2.5, alpha=1,
            label=f'{model}', markersize=10)
    # label=f'{model} (F={features[i]})', markersize=8)

    # 填充区域（半透明）
    ax.fill(angles, values, alpha=0.1, color=colors[i])

    # 外框加粗
    ax.spines['polar'].set_linewidth(2)  # 外框加粗

# 添加指标标签
# ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=20)

# # 指标标签偏移
# 添加指标标签
ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=24, weight="bold")

# 调整特定标签的位置
# 获取所有标签
labels = ax.get_xticklabels()

# 找到"AUC"标签的索引（通常是第三个标签，索引为2）
for i, label in enumerate(labels):
    if label.get_text() == "AUC":
        # 调整水平对齐方式为右对齐
        label.set_ha('right')
        # 微调位置（负值表示向左移动）
        label.set_position((label.get_position()[0] - 0.05, label.get_position()[1]))
        break

# 找到"Accuracy"标签的索引（通常是第三个标签，索引为2）
for i, label in enumerate(labels):
    if label.get_text() == "Accuracy":
        # 调整水平对齐方式为右对齐
        label.set_ha('left')
        # 微调位置（负值表示向左移动）
        label.set_position((label.get_position()[0] + 0.05, label.get_position()[1]))
        break

# 设置径向轴
ax.set_rlabel_position(45)
plt.yticks([0.85, 0.9, 0.95], ["0.85", "0.9", "0.95"], color="black", weight="bold", size=20)
plt.ylim(0.7, 0.95)

# 添加图例和标题
plt.legend(loc='upper right', bbox_to_anchor=(1.27, 1), frameon=True, shadow=False)

# 设置径向网格线
plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.7)

plt.tight_layout()
plt.savefig('radar_performance_pink.tif', dpi=300, bbox_inches='tight')
plt.show()

#Fig. 3c
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

# 定义 Optuna 的 objective 函数
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
    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    # 返回交叉验证的平均准确率
    return np.mean(scores)

# 使用 Optuna 进行超参数优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  # 运行100次试验

# 打印最佳超参数
best_params = study.best_params
print("最佳超参数：", best_params)

# 使用最佳超参数重新训练模型
best_model = CatBoostClassifier(
    depth=best_params['depth'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    random_state=0,
    verbose=0  # 禁用日志输出
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
n_features_to_select = rfecv.min_features_to_select
n_features_range = list(range(n_features_to_select, len(X.columns) + 1))
mean_test_scores = rfecv.cv_results_['mean_test_score']
# 找到最优特征数量对应的索引和值
index_optimal = n_features_range.index(len(selected_features))
value_optimal = mean_test_scores[index_optimal]
# 绘制特征数量与准确率的关系图
plt.figure(figsize=(10, 8))
fig = plt.figure()
plt.rcParams['font.family'] = 'Arial'
ax = fig.add_subplot(111)
ax.set(xlim=[4, 55], ylim=[0.75, 1])
plt.plot(n_features_range, mean_test_scores, marker='o', c="#1E88E5",label=None)
plt.xlabel('Features',fontsize=20, fontweight='bold')
plt.ylabel('Test accuracy',fontsize=20, fontweight='bold')
plt.tick_params(labelsize=16)
plt.axhline(y=0.9125, c='#FF0D57',linestyle='--',linewidth=2, markerfacecolor='w',markersize=10,zorder=0)
plt.axvline(x=54,ymax=0.66, c="#FF6F91",linestyle='--',linewidth=2,markerfacecolor='w',markersize=10,zorder=0)
plt.scatter(len(selected_features), value_optimal,c="#f5af02", marker='o', zorder=5, label="Optimal Model (n=54)")
plt.legend(loc="upper right",fontsize='xx-large',ncol=3)
plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
plt.gcf().subplots_adjust(left=0.14, bottom=0.13)
plt.savefig('.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()

#Fig.3d
#The method for drawing Fig. 3d is shown in SHAP.ipynb

#Fig. 4a
data = pd.read_csv("DES 58 3.csv")  # 替换为你的数据文件路径
label_column = data.columns[-1]  # 假设最后一列是标签列
X = data.drop(columns=[label_column])  # 特征集
y = data[label_column]  # 标签列

model = CatBoostClassifier(
depth=6, learning_rate=0.04768599666111316, n_estimators=113, l2_leaf_reg=5.29032308340007, random_state=0, verbose=0  # 禁用日志输出
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# 定义RFE对象
rfecv = RFECV(estimator=model, min_features_to_select=5, step=1, cv=skf, scoring='accuracy')  # 最小特征数量为5，每次消除1个特征
# 拟合RFE
rfecv.fit(X, y)

# 打印被选择的特征
selected_features = X.columns[rfecv.support_]
print("被选择的特征：", selected_features)

# 使用RFE选择的特征
X_rfecv = rfecv.transform(X)
predict_data = pd.read_csv("DES prediction(CAT).csv")  # 替换为你的新数据文件路径
# 假设第一列是序号
predict_index = predict_data.iloc[:, 0]  # 提取序号列
predict_X = predict_data.iloc[:, 1:]  # 提取特征列
# 确保新数据集的特征与训练数据集的特征一致
new_X = predict_data[selected_features]

# 使用训练好的模型进行预测
predictions = model.predict(new_X)
# 使用训练好的模型进行概率预测
probabilities = model.predict_proba(new_X.values)

# 提取属于类别 0 和类别 1 的概率
probabilities_class_0 = probabilities[:, 0]  # 类别 0 的概率
probabilities_class_1 = probabilities[:, 1]  # 类别 1 的概
# 计算前10%和后10%的数据范围
n = len(probabilities_class_1)
threshold_50 = np.percentile(probabilities_class_1, 50)  # 50%的阈值
# 绘制直方图
plt.figure(figsize=(12, 6))
plt.xlim([0.024, 1])
plt.ylim([0, 200])
# 设置字体和标签大小
plt.rcParams['font.family'] = 'Arial'
# 绘制0-50%和50%-100%的直方图
n1, bins1, patches1 = plt.hist(probabilities_class_1[(probabilities_class_1 >= 0.1) & (probabilities_class_1 < 0.5)],
         bins=18, color='#808080', alpha=0.5, edgecolor="white", label='Probability < 50%')
n2, bins2, patches2 = plt.hist(probabilities_class_1[(probabilities_class_1 >= 0.5) & (probabilities_class_1 <= 0.6)],
         bins=4, color='#F3B169', alpha=0.5, edgecolor="white", label='Probability > 50%')
# 绘制0.6的数据
n3, bins3, patches3 = plt.hist(probabilities_class_1[(probabilities_class_1 >= 0.6)], bins=18, color='#589FF3', alpha=0.5, edgecolor="white", label='Probability > 60%')
# 绘制后0.1的数据
n4, bins4, patches4 = plt.hist(probabilities_class_1[(probabilities_class_1 < 0.1)], bins=4, color='#F94141', alpha=0.5, edgecolor="white", label='Probability < 10%')

# 使用所有数据的直方图计数值绘制曲线
n, bins = np.histogram(probabilities_class_1, bins=44)
bin_centers = 0.5 * (bins[1:] + bins[:-1])  # 计算每个bin的中心点
X_Y_Spline = make_interp_spline(bin_centers, n)
X_ = np.linspace(0.024, 1, 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, color='black', linewidth=2, label='Probability Curve')

# 填充从曲线到柱子之间的面积
# 0-50%的区域
plt.fill_between(X_, Y_, color='#808080', alpha=0.3, where=np.logical_and(X_ < 0.5 , X_ >= 0.1))
plt.fill_between(X_, Y_, color='#FF0D57', alpha=0.3, where=(X_ < 0.1))
# 50%-100%的区域
plt.fill_between(X_, Y_, color='#F3B169', alpha=0.3, where=np.logical_and(X_ >= 0.5 , X_ < 0.6))
plt.fill_between(X_, Y_, color='#1E88E5', alpha=0.3, where=(X_ >= 0.6))
# 添加垂直区分线
plt.axvline(0.1, 0, 0.63, linewidth=4, color="#47AF79", label='Probability of CPL(10%)')
plt.axvline(0.5, 0, 0.02, linewidth=4, color="black", label='Probability threshold of models (50%)')
plt.axvline(0.6, 0, 0.05, linewidth=4, color="#e53238", label='Probability of CPL(60%)')
# 在垂直线上添加标签
#plt.text(bottom_10_percent, plt.ylim()[1] * 0.69, f'Bottom 10%: {bottom_10_percent:.2%}',
#         color="#0064d2", fontsize=14, verticalalignment='top', horizontalalignment='left')
#plt.text(top_10_percent, plt.ylim()[1] * 0.135, f'Top 10%: {top_10_percent:.2%}',
#        color="#e53238", fontsize=14, verticalalignment='top', horizontalalignment='center')
#plt.text(0.5, plt.ylim()[1] * 0.1, '50%', color="black", fontsize=14, verticalalignment='top', horizontalalignment='center')

plt.plot(0.965, 41, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.9433, 40, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.9233, 24, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.86, 18, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.84, 25, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2', label='No CPL')
plt.plot(0.84, 33, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.82, 18, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.695, 22, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.611, 26, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.611, 18, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238', label='CPL')
plt.plot(0.332, 26, marker='o', color='purple', markersize=12, markerfacecolor='#C99BFF', label='Special Point')
plt.plot(0.09, 183, marker='*', color='#FF0D57', markersize=12, markerfacecolor='#e53238')
plt.plot(0.09, 175, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.0689, 177, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.05, 190, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.05, 182, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.05, 174, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.05, 166, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.05, 158, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.05, 150, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
plt.plot(0.03, 80, marker='d', color='#1E88E5', markersize=10, markerfacecolor='#0064d2')
# 设置字体和标签大小
plt.rcParams['font.family'] = 'Arial'
plt.xlabel('Predictive probability of CPL', fontsize=20, fontweight='bold')
plt.ylabel('Number', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# 添加图例
plt.legend(fontsize=16,frameon=False)
# 显示图形
plt.tight_layout()
plt.savefig('.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()
#Fig. 4b
data = pd.read_csv("DES 58 3.csv")  # 替换为你的数据文件路径
label_column = data.columns[-1]  # 假设最后一列是标签列
X = data.drop(columns=[label_column])  # 特征集
y = data[label_column]  # 标签列

model = CatBoostClassifier(
depth=6,
learning_rate=0.04768599666111316,
n_estimators=113,
l2_leaf_reg=5.29032308340007,
random_state=0,
verbose=0  # 禁用日志输出
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# 定义RFE对象
rfecv = RFECV(estimator=model, min_features_to_select=5, step=1, cv=skf, scoring='accuracy')  # 最小特征数量为5，每次消除1个特征
# 拟合RFE
rfecv.fit(X, y)

# 打印被选择的特征
selected_features = X.columns[rfecv.support_]
print("被选择的特征：", selected_features)

# 使用RFE选择的特征
X_rfecv = rfecv.transform(X)
model.fit(X_rfecv, y)

feature_importances = model.feature_importances_  # 获取特征重要性
feature_names = selected_features

# 示例：加载新的数据集
predict_data = pd.read_csv("DES prediction(CAT).csv")  # 替换为你的新数据文件路径
# 假设第一列是序号
predict_index = predict_data.iloc[:, 0]  # 提取序号列
predict_X = predict_data.iloc[:, 1:]  # 提取特征列
# 确保新数据集的特征与训练数据集的特征一致
new_X = predict_data[selected_features]

# 使用训练好的模型进行预测
predictions = model.predict(new_X)
# 使用训练好的模型进行概率预测
probabilities = model.predict_proba(new_X.values)

# 提取属于类别 0 和类别 1 的概率
probabilities_class_0 = probabilities[:, 0]  # 类别 0 的概率
probabilities_class_1 = probabilities[:, 1]  # 类别 1 的概

# 计算每个概率的排名
ranks = np.argsort(np.argsort(probabilities_class_1)) + 1  # 从1开始排名
rank_percentages = (ranks / len(probabilities_class_1)) * 100  # 转换为百分比
# 绘制散点图
plt.figure(figsize=(6, 6))
plt.rcParams['font.family'] = 'Arial'
plt.xlim([-5, 105])
plt.ylim([0, 1])
plt.xticks([0, 20, 40, 60, 80, 100])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.scatter(rank_percentages, probabilities_class_1, alpha=0.6, marker='p', color="#e5e5e5", s=60, label='Candidate')
plt.ylabel("Probability of CPL", fontsize=20, fontweight='bold')
plt.xlabel("Rank percentage (%)",fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 添加垂直线（原水平线）
plt.axhline(y=0.6, xmin=0, xmax=1, color='#FF0D57', linewidth=3, label='Probability of CPL (60%)')
plt.axhline(y=0.1, xmin=0, xmax=1, linewidth=3, color="#1E88E5", label='Probability of CPL (10%)')
plt.axhspan(0.6, 1, xmin=0, xmax=1,  color='lightcoral', alpha=0.3)  # 填充排名78.99到100的区域
plt.axhspan(0, 0.1, xmin=0, xmax=1, color='lightblue', alpha=0.3)  # 填充概率0到0.1的区域
# 特定点的序号
blue_points_indices = [1207, 756, 642, 782, 31, 62, 1061, 881, 758, 258, 16]
red_points_indices = [693, 720, 774, 110, 724, 450, 776, 695, 235]
purple_points_indices = [185]
# 获取特定点的坐标
blue_points_prob = probabilities_class_1[predict_index.isin(blue_points_indices)]
blue_points_rank = rank_percentages[predict_index.isin(blue_points_indices)]
red_points_prob = probabilities_class_1[predict_index.isin(red_points_indices)]
red_points_rank = rank_percentages[predict_index.isin(red_points_indices)]
purple_points_prob = probabilities_class_1[predict_index.isin(purple_points_indices)]
purple_points_rank = rank_percentages[predict_index.isin(purple_points_indices)]
# 绘制特定点
plt.scatter(blue_points_rank, blue_points_prob, alpha=1.0, marker='x', color='#1E88E5', s=80, label='No CPL')
plt.scatter(red_points_rank, red_points_prob, alpha=1.0, marker='o', color='#FF0D57', s=80, label='CPL')
plt.scatter(purple_points_rank, purple_points_prob, alpha=1.0, marker='o', color='#C99BFF', s=80, label='Special Point')
# 在垂直线上添加标签
#plt.text(0.9, 6, 'y=5.34%',color="#6888F5", fontsize=14, verticalalignment='bottom', horizontalalignment='center')
#plt.text(0.9, 79, 'y=80.68%',color="red", fontsize=14, verticalalignment='top', horizontalalignment='center')
# 添加图例
plt.legend(fontsize=14, frameon=False, bbox_to_anchor=(0.02, 0.58), loc='upper left')
plt.savefig('.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()