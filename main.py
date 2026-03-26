from src.utils import *
from src.feature_engineering import *
from src.model_training import *
import os

# 读取数据
print('正在读取数据...')
data = read_data()

# 分离出训练集标签
target = data['train']['TARGET']
data['train'].drop('TARGET', axis=1, inplace=True)
print('已分离出训练集标签！')

# 查看缺失值分布
# nan_distribution(data)

# 合并所有表
print('开始合并所有表')
train, test = table_merge(data)
print('合并完成！')

# 分离出测试集ID
curr_id = test['SK_ID_CURR']
test.drop(['SK_ID_CURR'], axis=1, inplace=True)
train.drop(['SK_ID_CURR'], axis=1, inplace=True)
# train.info(verbose=True)
print('测试集ID已被分离')

# 填充缺失值
print('正在填充缺失值')
train, test = fill(train, test)
# 验证是否填充完毕
verify_fill(train)
verify_fill(test)

# 特征衍生
print('正在进行特征衍生')
train, test = add_features(train, test)
print('特征衍生完毕！')

# 再次填充
train, test = fill(train, test)
verify_fill(train)
verify_fill(test)

# 类别编码：encoder可选'label'或'onehot'
print('正在进行类别编码')
train, test = cat_feature_encoder(train, test,encoder='onehot')
print('类别编码完成！')

# 再次挤压内存
print('再次挤压内存')
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# 特征选择
print('进行特征选择')
train_reduced, test_reduced = feature_selection(train, test,target)

# 对LGBM进行随机搜索调参（耗时操作，方案是拿到参数后注释调参代码，重新实例化模型）
# lgbm_HPO(train_reduced,target)
# 最好的参数：{'learning_rate': np.float64(0.08169314570885453), 'max_depth': 3, 'num_leaves': 57, 'reg_alpha': np.float64(0.8631034258755935), 'reg_lambda': np.float64(0.6232981268275579)}
# 最好的分数：0.7743393515494559

# 不能将见过训练数据的model传入集成模型，必须拿到调好的参数重新实例化一个模型
new_main_model = LGBMClassifier(
    class_weight='balanced',
    metric='roc_auc',
    random_state=42,
    n_jobs=-1,
    n_estimators=1000,
    learning_rate=0.08,
    max_depth=3,
    num_leaves=57,
    reg_alpha=0.86,
    reg_lambda=0.62
)

# KNN与SVM调参
# clean_knn_params,clean_svm_params = knn_svc_HPO(train_reduced,target)

# 集成模型所需的管道
knn_pipeline, svm_pipeline, lr_pipeline = model_pipeline()

# 实例化集成模型
print('训练集成模型')
stacking = model_ensemble(evaluate=False,X=train_reduced,y=target,new_main_model=new_main_model,knn_pipeline=knn_pipeline,lr_pipeline=lr_pipeline)

# 集成模型和LGBM分别预测拿到结果
if __name__ == '__main__':
    # 集成模型预测
    proba1 = stacking.predict_proba(test_reduced)[:, 1]
    stacking_result = pd.DataFrame({
        'SK_ID_CURR': curr_id,
        'TARGET': proba1
    })
    print('集成模型预测完毕！')

    # 只用LGBM预测
    new_main_model.fit(train_reduced, target)
    proba2 = new_main_model.predict_proba(test_reduced)[:, 1]
    lgb_result = pd.DataFrame({
        'SK_ID_CURR': curr_id,
        'TARGET': proba2
    })
    print('lgbm预测完毕！')

    # 定义输出目录
    output_dir = 'outputs'

    # 如果目录不存在，就创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    else:
        print(f'检测到目录已存在: {output_dir}/ (将覆盖原有结果文件)')

    # 保存文件时，指定路径到 outputs 文件夹
    stacking_result.to_csv(os.path.join(output_dir, 'sub_stacking.csv'), index=False)
    lgb_result.to_csv(os.path.join(output_dir, 'sub_LGB.csv'), index=False)

    print(f"预测结果已保存至 {output_dir}/ 目录")
