[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green.svg)](https://lightgbm.readthedocs.io/)
## 项目简介
本项目基于 Home Credit Default Risk 数据集，构建机器学习模型预测贷款申请人的违约风险。通过整合多个数据表（申请信息、历史信贷记录、分期还款记录等），进行特征工程、模型调参和集成学习，最终输出违约概率预测结果。

## 项目目标
预测贷款申请人是否会违约（二分类问题）

评估指标：AUC-ROC

输出：每个申请人的违约概率
## 🌟 项目亮点

- ✅ **内存优化**：自定义 `reduce_mem_usage` 函数，内存占用降低 70%+
- ✅ **多表融合**：整合 7 个数据表，构建 50+ 业务特征
- ✅ **集成学习**：Stacking 融合 5 个基模型，提升泛化能力
- ✅ **完整 Pipeline**：从数据清洗到模型部署的全流程实现
- ✅ **可复现**：固定随机种子，代码结构清晰，便于扩展
  

## 数据说明

| 数据表 | 说明 |
|--------|------|
| `application_train.csv` | 训练集（包含标签TARGET） |
| `application_test.csv` | 测试集 |
| `bureau.csv` | 客户在其他金融机构的信贷记录 |
| `bureau_balance.csv` | 信贷记录月度余额 |
| `previous_application.csv` | 客户历史贷款申请记录 |
| `POS_CASH_balance.csv` | POS现金贷款余额 |
| `installments_payments.csv` | 分期还款记录 |
| `credit_card_balance.csv` | 信用卡余额记录 |


## 环境要求
Python >= 3.8
### 依赖包
- numpy
- pandas
- matplotlib
- scikit-learn
- lightgbm
- xgboost
- scipy
- gc
### 安装命令
pip install -r requirements.txt
## 项目结构
project/

├── main.py                 # 主程序代码

├── src/                    # 源代码模块

│       ├── feature_engineering.py

│       ├── model_training.py

│       └── utils.py

├── notebooks/              # Jupyter实验笔记

├── requirements.txt        # 依赖包列表

├── README.md               # 项目说明

└── outputs/                # 预测结果输出目录（.gitignore忽略）

⚠️ 注意：数据文件未包含在仓库中，需自行从 Kaggle 下载

## 代码模块说明
### 1. 数据读取与内存优化
```python
reduce_mem_usage(df)  # 在保证不溢出的前提下降低数据类型精度，节省内存

read_data()           # 读取所有CSV文件
```

- 可自动检测数值列范围，降级为更小的数据类型（int8/16/32, float16/32）

- 类别列转为 category 类型

- 内存减少约 70-80%

### 2. 多表合并与聚合
```python
table_merge(dic)  # 合并所有数据表
```
合并逻辑

bureau_balance → bureau → train/test

POS_CASH/Installments/Credit_Card → previous → train/test


**聚合策略：**

- 数值列：min, max, mean, sum, std, p25, p75

- 类别列：count, nunique

**新增衍生特征：**

逾期天数、还款比例、逾期标记等

### 3. 缺失值处理
```python
nan_distribution(dic) # 查看缺失值分布
fill(df, test_n)  # 填充缺失值
verify_fill(df) # 检查缺失值是否填充完毕
```
| 特征类型 | 缺失率 | 处理策略 |
|---------|--------|---------|
| 数值型	 | <10%	| 中位数填充 |
| 数值型	 | 10%-80% | 最小值-1填充 |
| 数值型	 | >80%	| 删除该列 |
| 类别型	 | <10%	| 众数填充 |
| 类别型 | >10% |	填充'UNKNOWN' |

> nan_distribution反映了缺失值对于该特征的占比，选择了10%作为分界线；
> 大于80%则学不到东西，大于10%如果强行填充可能损坏数据原始分布，因此都设计了缺失标记

### 4. 特征工程
```python
add_features(train, test)  # 添加衍生特征
```
#### 主要衍生特征：

| 类别 | 特征示例 |
|------|----------|
| 比率特征 | `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO` |
| 外部评分 | `EXT_SOURCE_MEAN`, `EXT_SOURCE_STD` |
| 稳定性 | `STABILITY_SCORE`, `EMPLOYED_YEARS` |
| 债务负担 | `DEBT_BURDEN_SCORE`, `HIGH_RISK_SCORE` |
| 年龄相关 | `AGE_YEARS`, `AGE_BIN`, `EMPLOYED_AGE_RATIO` |
| 资产相关 | `HAS_CAR`, `HAS_REALTY`, `ASSET_COUNT` |
| 缺失标记 | `TOTAL_MISSING_COUNT`, `MISSING_RATIO` |


### 5. 类别特征编码
```python
cat_feature_encoder(train, test, *,encoder)  # encoder可选'label'或'onehot'
```
### 6. 特征选择
```python
feature_selection(train,test,target,fig=False)  # 基于LGBM特征重要性筛选,fig=True可展示前20重要性特征
```
- 使用 LightGBM.feature_importences_拿到特征重要性
- 保留使用次数 > 20 的特征（n_estimators=1000）
- 可视化 Top 20 重要特征
### 7. 模型调参
```python
lgbm_HPO(X,y)      # LightGBM 随机搜索调参
knn_svc_HPO(X,y)   # KNN & SVM 网格搜索调参
```
#### LGBM 最优参数：
```python
{
    'learning_rate': 0.08,
    'max_depth': 3,
    'num_leaves': 57,
    'reg_alpha': 0.86,
    'reg_lambda': 0.62
}
```
### 8. Stacking 集成模型
```python
model_ensemble(X,y,new_main_model,knn_pipeline,lr_pipeline,evaluate=True,svm_pipeline=None)  # 构建Stacking集成,svm默认不使用
```
**基学习器：**

- XGBoost Classifier
- LightGBM Classifier
- Random Forest Classifier
- Gaussian Naive Bayes
- KNN (Pipeline with StandardScaler)
- (设计了SVM管道，本意是为集成模型增加视角，但由于数据量问题没有使用)
  
**元学习器：**

Logistic Regression

> 默认迭代次数不足以收敛，代码中max_iter=3000

**交叉验证：** StratifiedKFold (n_splits=2)
### 9. 预测与输出
```python
#集成模型预测
stacking.predict_proba(test_reduced)

#单模型预测
new_main_model.predict_proba(test_reduced)
```
#### 输出文件：
- `sub_stacking.csv` - 集成模型预测结果
- `sub_LGB.csv` - LightGBM单模型预测结果
## 使用方法
### 1. 准备数据
请从 Kaggle Home Credit 竞赛页面下载以下文件放入项目根目录：

1. application_train.csv
2. application_test.csv
3. bureau.csv
4. bureau_balance.csv
5. previous_application.csv
6. POS_CASH_balance.csv
7. installments_payments.csv
8. credit_card_balance.csv
### 2. 运行代码
python main.py

*代码中各模型的n_jobs设置在<=4，可防止进程错误，若内存充足（>32G）可将n_jobs设为 -1 加速训练*
### 3. 查看结果
**预测结果将保存为：**
- `sub_stacking.csv`
- `sub_LGB.csv`
> 代码中自动生成outputs目录并将输出文件保存于此，但被 .gitignore 忽略
 
## 模型性能

| 模型 | AUC-ROC |
|------|---------|
| LightGBM (单模型) | 0.771 ± 0.002 |
| Stacking 集成 | 0.772 ± 0.001 |

> 实际性能取决于数据质量和调参结果
## 注意事项
内存占用：

- 完整数据合并时可能占用较大内存，建议至少 16GB RAM
- 编码后建议再次调用内存函数（reduce_mem_usage）


运行时间：

打开网格cv：~1.5-3小时

关闭网格cv：~5-20分钟

> 注：SVM 在大规模数据上训练时间较长（>12小时），性价比低，最终未纳入集成模型。建议在生产环境中优先选择 LightGBM/XGBoost 等高效算法，但一定要注意视角多样性，多个同类算法可能学习到相同的噪声，导致错误被放大。
         
随机种子：已设置 random_state=42 保证可复现性

调参注释：部分调参代码已注释，如需重新调参请取消注释
## 改进方向
1. **深度学习时序建模**：尝试 TabNet/Transformer 捕捉 `installments_payments` 等时序表的长期依赖
2. **自动化特征工程**：使用 FeatureTools 自动生成更多交叉特征
3. **模型解释性**：集成 SHAP 值分析，提升模型业务可解释性
4. **API 部署**：使用 FastAPI 封装模型，实现实时预测服务

4. **处理类别不平衡问题（SMOTE, 欠采样等）**
## 许可证


## 联系方式
如有问题或建议，请提交 Issue 。 

## 致谢

数据集来源：Kaggle - Home Credit Default Risk

参考方案：Kaggle竞赛优秀Notebook

**感谢！**




