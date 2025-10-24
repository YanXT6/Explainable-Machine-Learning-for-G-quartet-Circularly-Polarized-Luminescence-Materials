import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("DES 58.csv")  # 替换为你的原始数据文件路径

# 计算每一列的标准差，如果标准差为0，则该列的所有值都相同
constant_columns = data.columns[data.std() == 0]
data = data.drop(columns=constant_columns)
# 计算阈值，即原始数据列数的80%
Thresh = int(0.8 * len(data))

# 删除缺失值过多的列（缺失值超过80%的列）
X_NAomit_data = data.dropna(
    thresh=Thresh,  # 保留至少有Thresh个非空值的列
    axis=1  # 按列操作
)

# 4. 删除95%数据相同的列
def check_highly_similar_columns(df, threshold=0.95):
    columns_to_drop = []
    for col in df.columns:
        # 计算该列中每个值的出现频率
        value_counts = df[col].value_counts(normalize=True)
        # 如果某个值的频率超过阈值，则认为该列数据高度相似
        if value_counts.max() > threshold:
            columns_to_drop.append(col)
    return columns_to_drop

# 检查并删除95%数据相同的列
highly_similar_columns = check_highly_similar_columns(data, threshold=0.93)
data = data.drop(columns=highly_similar_columns)

# 打印处理后的数据的行列数
print(f"处理后的数据行列数：{data.shape}")
# 检查处理后的数据是否仍然存在缺失值
missing_values = data.isnull().sum().sum()
if missing_values > 0:
    print(f"警告：处理后的数据中仍然存在 {missing_values} 个缺失值。")
else:
    print("处理后的数据中没有缺失值。")

# 报告每列中缺失值的数量
missing_values_per_column = data.isnull().sum()
columns_with_missing_values = missing_values_per_column[missing_values_per_column > 0]
if not columns_with_missing_values.empty:
    print("\n以下列包含缺失值：")
    print(columns_with_missing_values)
else:
    print("\n没有列包含缺失值。")

# 将处理后的数据保存为CSV文件
output_path = "DES 58 1.csv"  # 输出文件路径
data.to_csv(output_path, sep=',', header=True, index=False)

print(f"处理后的数据已保存到：{output_path}")

data = pd.read_csv("DES 58 1.csv")
print(data.head(58))
print("数据总行数：", len(data))
# 第一列是标签列，其余列是特征列
label_column = data.columns[-1]  # 标签列名
feature_columns = data.columns[1:]  # 特征列名列表

# 初始化一个空的DataFrame来存储Wilcoxon秩和检验的结果
results = []
# 分组：第2-4行分为一组，其他行分为另一组
group1_indices = list(range(0, 20))
group1 = data.iloc[group1_indices]  # 提取group1的数据
group2_indices = list(range(20, 58))
group2 = data.iloc[group2_indices]  # 提取group2的数据
print(group1)
print(group2)
# 遍历每一列特征进行Wilcoxon秩和检验
for feature in feature_columns:
    # 根据标签列分组# 提取每组的特征数据
    group1_data = group1[feature].dropna()
    group2_data = group2[feature].dropna()

    # 确保每个组至少有2个样本
    if len(group1_data) < 2 or len(group2_data) < 2:
        print(f"Skipping feature {feature} due to insufficient data in one of the groups.")
        continue

        # 计算倍数变化（Fold Change）
    mean_group1 = group1_data.mean()
    mean_group2 = group2_data.mean()
    fold_change = mean_group1 / mean_group2
    log_fc = np.log(fold_change)  # 对数倍数变化

    # 进行Wilcoxon秩和检验
    stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')

    # 将结果添加到结果DataFrame中
    results.append({'Feature': feature, 'P-value': p_value, 'Log2 Fold Change': log_fc})

# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results).set_index('Feature')

# 对结果按P-value进行排序
results_sorted = results_df.sort_values(by='P-value')

# 筛选出P-value小于0.05的结果
significant_results = results_sorted[results_sorted['P-value'] < 0.05]

# 打印处理后的数据的行列数
print("处理后的数据行列数：")
print(f"行数: {significant_results.shape[1]}")
print(f"列数: {significant_results.shape[1]}")

# 保存到新的CSV文件
output_file_path = 'DES P 58.csv'  # 输出文件路径
significant_results.to_csv(output_file_path)

print(f"Significant Wilcoxon rank-sum test results saved to {output_file_path}")

# 提取显著特征的原始数据
significant_features = significant_results.index.tolist()
significant_data = data[[label_column] + significant_features]

# 保存显著特征的原始数据到一个新的CSV文件
significant_data_output_path = 'DES 58 2.csv'
significant_data.to_csv(significant_data_output_path, index=False)

print(f"Significant features' original data saved to {significant_data_output_path}")

# 计算负对数P值
results_sorted['-log P-value'] = -np.log(results_sorted['P-value'])

plt.figure(figsize=(5, 5)) # 绘制火山图
# 设置绘图风格
sns.scatterplot(x='Log2 Fold Change', y='-log P-value', data=results_sorted, hue=(results_sorted['P-value'] < 0.05), palette=['gray', 'red'], legend=False)
# 定义颜色映射
cm2 = plt.colormaps['BuPu_r']
results_sorted_a = results_sorted[results_sorted['P-value'] < 0.05]
results_sorted_b = results_sorted[results_sorted['P-value'] >= 0.05]
# 绘制散点图
sc1 = plt.scatter(results_sorted_a['Log2 Fold Change'], results_sorted_a['-log P-value'], c="#e53238", cmap=cm2, label='Group A')
sc2 = plt.scatter(results_sorted_b['Log2 Fold Change'], results_sorted_b['-log P-value'], c="#d6dddf", cmap=cm2, label='Group B')

ax=plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.20)
plt.axhline(y=-np.log(0.05),c="#0064d2",linestyle='--',linewidth=2,markerfacecolor='w',markersize=10)
plt.tick_params(labelsize=16)
ax.set_xlabel('log FC',fontsize=16)
ax.set_ylabel('log P value',fontsize=16)
plt.xlim(-6, 6)
plt.savefig(".tiff", dpi=300, format="tiff", bbox_inches='tight')
plt.show()

data = pd.read_csv("DES 58 2.csv")  # 替换为你的原始数据文件路径
# 假设data是你的DataFrame
label_column = data.columns[-1]  # 假设最后一列是标签列

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=[label_column]))
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[1:])
data_scaled[label_column] = data[label_column]  # 将标签列重新加入

# 计算相关性矩阵
corr_matrix = data_scaled.corr(method='spearman')

# 找出高相关性的特征对
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# 删除高相关性的特征
data_cleaned = data.drop(columns=to_drop)

print("清理后的数据形状：", data_cleaned.shape)
# 保存到新的CSV文件
output_file_path = 'DES 58 3.csv'  # 输出文件路径
data_cleaned.to_csv(output_file_path, sep=',', header=True, index=False)

# 重新计算清理后的数据的相关性矩阵
corr_matrix_cleaned = data_cleaned.drop(columns=[label_column]).corr(method='spearman')
#corr_matrix_cleaned = data_cleaned.corr(method='spearman')
print(f"清理后的数据已保存到 {output_file_path}")

# 绘制相关性热图
#plt.figure(figsize=(9, 8))
#sns.clustermap(corr_matrix, vmax=1, vmin=-1, center=0, cmap='RdBu_r', annot=False)
#plt.tight_layout()
# 保存热图
#clustermap_output_path = '.tiff'
#plt.savefig(clustermap_output_path, dpi=300, format="tiff", bbox_inches='tight')
#print(f"相关性热图已保存到 {clustermap_output_path}")
# 显示热图
#plt.show()

# 绘制相关性热图
plt.figure(figsize=(9, 8))
sns.clustermap(corr_matrix_cleaned, vmax=1, vmin=-1, center=0, cmap='RdBu_r', annot=False)
plt.title("Correlation Heatmap", fontsize=12)
plt.tight_layout()
# 保存热图
clustermap_output_path1 = '.tiff'
plt.savefig(clustermap_output_path1, dpi=300, format="tiff", bbox_inches='tight')
print(f"相关性热图已保存到 {clustermap_output_path1}")
# 显示热图
plt.show()