import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import gc
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# 连接所有表
def table_merge(dic):
    """
    bureau_balance['SK_ID_BUREAU'] ——> bureau
    bureau['SK_ID_CURR'] ——> train/test

    POS_CASH_balance['SK_ID_PREV'] ——> previous
    installments_payments['SK_ID_PREV'] ——> previous
    credit_card_balance['SK_ID_PREV'] ——> previous
    previous['SK_ID_CURR'] ——> train/test
    """

    def bureau_balance_agg():
        """
        对bureau_balance进行分组聚合，以防止笛卡尔积爆炸
        :return: agg_bureau_balance
        """
        agg_dict = {}
        for col in dic['bureau_balance'].columns:
            if col == 'SK_ID_CURR' or col == 'SK_ID_BUREAU':
                continue
            elif dic['bureau_balance'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_bureau_balance = dic['bureau_balance'].groupby('SK_ID_BUREAU').agg(agg_dict)
        # 扁平化列名
        agg_bureau_balance.columns = ['bureau_balance_' + '_'.join(col).upper() for col in agg_bureau_balance.columns]
        # 重置索引
        agg_bureau_balance = agg_bureau_balance.reset_index()
        return agg_bureau_balance

    print('正在合并bureau_balance')
    agg_bureau_balance = bureau_balance_agg()
    bureau = pd.merge(dic['bureau'], agg_bureau_balance, how='left', on='SK_ID_BUREAU')
    # 清理agg_bureau_balance
    del agg_bureau_balance
    gc.collect()

    def bureau_agg():
        agg_dict = {}
        for col in bureau.columns:
            if col == 'SK_ID_CURR' or col == 'SK_ID_BUREAU':
                continue
            elif bureau[col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_bureau = bureau.groupby('SK_ID_CURR').agg(agg_dict)
        # 扁平化列名
        agg_bureau.columns = ['bureau_' + '_'.join(col).upper() for col in agg_bureau.columns]
        # 重置索引
        agg_bureau = agg_bureau.reset_index()
        return agg_bureau

    agg_bureau = bureau_agg()
    # 将agg_bureau合并入训练测试集
    train_temp = pd.merge(dic['train'], agg_bureau, how='left', on='SK_ID_CURR')
    test_temp = pd.merge(dic['test'], agg_bureau, how='left', on='SK_ID_CURR')

    # 聚合POS_CASH_balance表
    def POS_CASH_balance_agg():
        agg_dict = {}
        for col in dic['POS_CASH_balance'].columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue
            elif dic['POS_CASH_balance'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_POS_CASH_balance = dic['POS_CASH_balance'].groupby('SK_ID_PREV').agg(agg_dict)
        agg_POS_CASH_balance.columns = ['POS_CASH_balance_' + '_'.join(col).upper() for col in
                                        agg_POS_CASH_balance.columns]
        agg_POS_CASH_balance = agg_POS_CASH_balance.reset_index()
        return agg_POS_CASH_balance

    agg_POS_CASH_balance = POS_CASH_balance_agg()
    print('正在合并previous0')
    previous0 = pd.merge(dic['previous'], agg_POS_CASH_balance, how='left', on='SK_ID_PREV')
    del agg_POS_CASH_balance

    # 聚合installments表
    def installments_agg():
        agg_dict = {}
        for col in dic['installments'].columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue
            elif dic['installments'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_installments = dic['installments'].groupby('SK_ID_PREV').agg(agg_dict)
        # 计算每次分期的逾期天数（应还日期 - 实际还款日期）
        # 负数表示逾期，正数表示提前还款
        dic['installments']['OVERDUE_DAYS'] = dic['installments']['DAYS_INSTALMENT'] - dic['installments'][
            'DAYS_ENTRY_PAYMENT']

        # 计算每次分期的还款比例
        dic['installments']['PAYMENT_RATIO'] = dic['installments']['AMT_PAYMENT'] / dic['installments'][
            'AMT_INSTALMENT'].replace(0, np.nan)

        # 标记是否逾期（逾期天数 > 0 表示逾期）
        dic['installments']['IS_OVERDUE'] = (dic['installments']['OVERDUE_DAYS'] > 0).astype(int)

        # 对新增字段做聚合
        overdue_agg = dic['installments'].groupby('SK_ID_PREV').agg({
            'OVERDUE_DAYS': ['max', 'mean', 'sum', 'std'],
            'PAYMENT_RATIO': ['min', 'mean', 'sum'],
            'IS_OVERDUE': ['sum', 'mean', 'count'],
        })

        # 合并原有聚合和逾期聚合
        agg_installments = agg_installments.merge(overdue_agg, how='left', left_index=True, right_index=True)
        agg_installments.columns = ['installments_' + '_'.join(col).upper() for col in agg_installments.columns]
        agg_installments = agg_installments.reset_index()
        return agg_installments

    agg_installments = installments_agg()
    print('正在合并previous1')
    previous1 = pd.merge(previous0, agg_installments, how='left', on='SK_ID_PREV')
    del previous0, agg_installments
    gc.collect()

    # 聚合credit_card_balance表
    def credit_card_balance_agg():
        agg_dict = {}
        for col in dic['credit_card_balance'].columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue
            elif dic['credit_card_balance'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_credit_card_balance = dic['credit_card_balance'].groupby('SK_ID_PREV').agg(agg_dict)
        agg_credit_card_balance.columns = ['credit_card_balance_' + '_'.join(col).upper() for col in
                                           agg_credit_card_balance.columns]
        agg_credit_card_balance = agg_credit_card_balance.reset_index()
        return agg_credit_card_balance

    agg_credit_card_balance = credit_card_balance_agg()
    print('正在合并previous2')
    previous2 = pd.merge(previous1, agg_credit_card_balance, how='left', on='SK_ID_PREV')
    del previous1, agg_credit_card_balance
    gc.collect()

    # 聚合previous2
    def previous_agg():
        agg_dict = {}
        for col in previous2.columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue
            elif previous2[col].dtype == np.number:
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:
                agg_dict[col] = ['count', 'nunique']
        agg_previous = previous2.groupby('SK_ID_CURR').agg(agg_dict)
        # 扁平化列名
        agg_previous.columns = ['previous_' + '_'.join(col).upper() for col in agg_previous.columns]
        # 重置索引
        agg_previous = agg_previous.reset_index()
        return agg_previous

    agg_previous = previous_agg()
    del previous2
    gc.collect()

    # 合并进训练测试集
    print('正在合并最后的训练测试集')
    train = pd.merge(train_temp, agg_previous, how='left', on='SK_ID_CURR')
    test = pd.merge(test_temp, agg_previous, how='left', on='SK_ID_CURR')
    del train_temp, test_temp, agg_previous
    gc.collect()

    # 清理原始字典大表
    del dic['bureau'], dic['bureau_balance'], dic['POS_CASH_balance']
    del dic['installments'], dic['credit_card_balance'], dic['previous']
    gc.collect()

    return train, test

# 填充缺失值
def fill(df, test_n):
    """
    数值型特征：缺失值大于10%的用-1填充，大于80%的删掉，少量缺失使用中位数填充
    类别型特征：缺失值小于10%的使用众数填充，否则使用UNKNOWN填充
    :return: df,test
    """
    from sklearn.impute import SimpleImputer

    # 数值特征列表与类别特征列表
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype('object')
    for col in test_n.select_dtypes(include=['category']).columns:
        test_n[col] = test_n[col].astype('object')
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    # 数值型缺失特征分类
    num_low_miss = []
    num_midian_miss = []
    num_high_miss = []
    for col in num_features:
        miss = df[col].isna().sum() / df.shape[0]
        if miss < 0.1:
            num_low_miss.append(col)
        elif 0.1 <= miss < 0.8:
            num_midian_miss.append(col)
        else:
            num_high_miss.append(col)

    # 类别特征中缺失值大于10%的特征
    cat_high_miss = []
    cat_low_miss = []
    for col in cat_features:
        if df[col].isna().sum() / df.shape[0] > 0.1:
            cat_high_miss.append(col)
        else:
            cat_low_miss.append(col)
    # 填充器
    if num_features:
        if num_low_miss:  # 少量缺失用中位数填充
            num_imputer = SimpleImputer(strategy='median')
            df[num_low_miss] = num_imputer.fit_transform(df[num_low_miss])
            test_n[num_low_miss] = num_imputer.transform(test_n[num_low_miss])
        if num_midian_miss:  # 中等缺失用‘列的最小值-1’填充
            for col in num_midian_miss:
                col_min = df[col].min()
                fill_value = col_min - 1 if not np.isnan(col_min) else -999
                df[col] = df[col].fillna(fill_value)
                test_n[col] = test_n[col].fillna(fill_value)
        if num_high_miss:  # 大量缺失删掉
            for col in num_high_miss:
                df = df.drop(col, axis=1)
                test_n = test_n.drop(col, axis=1, errors='ignore')
    if cat_low_miss:  # 少量缺失用众数填充
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_low_miss] = cat_imputer.fit_transform(df[cat_low_miss])
        test_n[cat_low_miss] = cat_imputer.transform(test_n[cat_low_miss])
    if cat_high_miss:  # 大量缺失用UNKNOWN
        for col in cat_high_miss:
            df[col] = df[col].fillna('UNKNOWN')
            test_n[col] = test_n[col].fillna('UNKNOWN')
    return df, test_n

# 验证是否填充完毕
def verify_fill(df):
    aa = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            aa.append(col)
    if len(aa) > 0:
        print(f"未填充完毕！未填充列表:{aa}，长度{len(aa)}")
    else:
        print("填充完毕！")

# 特征衍生
def add_features(train, test):
    for df in [train, test]:
        # 该项目中df['DAYS_EMPLOYED']特征如果值为365243，表示缺失值，需要特殊处理
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

        # 贷款/收入比
        df['CREDIT_INCOME_RATIO'] = np.log1p(df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL']+1))

        # 年金/收入比
        df['ANNUITY_INCOME_RATIO'] = np.log1p(df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL']+1))

        # 年金/商品价格比
        df['AMT_ANNUITY_GOODS_RATIO'] = np.log1p(df['AMT_ANNUITY'] / (df['AMT_GOODS_PRICE']+1))

        # 收入/商品价格比
        df['INCOME_GOODS_RAITO'] = np.log1p(df['AMT_INCOME_TOTAL'] / (df['AMT_GOODS_PRICE']+1))

        # 贷款/商品价格比
        df['CREDIT_GOODS_RATIO'] = np.log1p(df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE']+1))

        # 贷款金额与年金比值
        df['CREDIT_ANNUITY_RATIO'] = np.log1p(df['AMT_CREDIT'] / (df['AMT_ANNUITY']+1))

        # 人均收入
        df['INCOME_PER_PERSON'] = np.log1p(df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS']+1))

        # 年收入对数变换
        df['INCOME_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])  # log(1+x) 避免 log(0)

        # 年收入平方根变换
        df['INCOME_SQRT'] = np.sqrt(df['AMT_INCOME_TOTAL'])

        # 强特征交互
        df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] + df['EXT_SOURCE_2']
        df['EXT_SOURCE_2_3_add'] = df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']
        df['EXT_SOURCE_2_3_mult'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['EXT_SOURCE_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
        df['EXT_SOURCE_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
        df['EXT_SOURCE_RANGE'] = df['EXT_SOURCE_MAX'] - df['EXT_SOURCE_MIN']

        # 证件时效
        df['ID_DAYS_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']  # 证件发布相对年龄
        df['ID_DAYS_RATIO'] = df['ID_DAYS_RATIO'].replace([np.inf, -np.inf], 0).fillna(0)

        # 稳定性评分
        df['EMPLOYMENT_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

        # 金额比率
        df['ANNUITY_PER_PERSON'] = df['AMT_ANNUITY'] / (df['INCOME_PER_PERSON'] + 1)

        # 人口密度分箱
        density_labels = ['低密度', '中密度', '高密度']
        df['RPR_BIN'] = pd.qcut(
            x=df['REGION_POPULATION_RELATIVE'],
            q=[0, 0.33, 0.66, 1],
            labels=density_labels,
            duplicates='drop'
        )

        # 贷款金额异常高
        df['CREDIT_OUTLIER'] = (df['AMT_CREDIT'] > train['AMT_CREDIT'].quantile(0.99)).astype(int)

        # 受雇时长是否缺失
        df['DAYS_EMPLOYED_MISSING'] = df['DAYS_EMPLOYED'].isna().astype(int)

        # 收入是否缺失
        df['AMT_INCOME_TOTAL_MISSING'] = df['AMT_INCOME_TOTAL'].isna().astype(int)

        # 缺失值总数（每个样本的缺失列数）
        df['TOTAL_MISSING_COUNT'] = df.isna().sum(axis=1)

        # 缺失值比例
        df['MISSING_RATIO'] = df.isna().sum(axis=1) / df.shape[1]

        # 收入 × 工作年限 (收入稳定性)
        df['INCOME_EMPLOYMENT_RATIO'] = np.log1p(df['AMT_INCOME_TOTAL'] / (df['DAYS_EMPLOYED'].abs() + 1))

        # 收入分箱
        income_bins = train['AMT_INCOME_TOTAL'].quantile([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]).tolist()
        income_labels = ['极低收入','低收入','中低收入','中下收入','中上收入','中高收入','高收入','极高收入']
        income_bins[0] = -np.inf
        income_bins[-1] = np.inf
        df['INCOME_BIN'] = pd.cut(df['AMT_INCOME_TOTAL'],
                          bins=income_bins,
                          labels=income_labels,
                          right=True)

        # 工作开始的年龄
        df['WORK_START_AGE'] = np.clip(((df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']) / -365),0,None)

        # 工作开始年龄是否>35
        df['WORK_START_AGE_35'] = (df['WORK_START_AGE'] >= 35).astype(int)

        # 年龄（年）
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

        # 年龄是否大于35
        df['AGE_YEARS_35'] = (df['AGE_YEARS'] >= 35).astype(int)

        # 年龄异常
        df['AGE_OUTLIER'] = ((df['AGE_YEARS'] < 18) | (df['AGE_YEARS'] > 70)).astype(int)

        # 年龄分箱
        age_bins = [-np.inf, 18, 25, 40, 60, 70, np.inf]
        age_labels = ['过小','青年','成年','中年','老年','过老']
        df['AGE_BIN'] = pd.cut(df['AGE_YEARS'], bins=age_bins, labels=age_labels, right=True)

        # 工作年限（年）
        df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365

        # 注册时长（年）
        df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365

        # 身份证明变更时长（年）
        df['ID_CHANGE_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365

        # 原始变量（二值化）
        df['HAS_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
        df['HAS_REALTY'] = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
        df['HAS_CHILD'] = (df['CNT_CHILDREN'] > 0).astype(int)

    return train, test

# 类别特征编码
def cat_feature_encoder(train, test,*,encoder):
    if encoder == 'label':
        from sklearn.preprocessing import LabelEncoder
        cat_feature = train.select_dtypes(include=['object','category']).columns
        for col in cat_feature:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])
            test[col] = le.transform(test[col])
    if encoder == 'onehot':
        from sklearn.preprocessing import OneHotEncoder

        cat_feature = train.select_dtypes(include=['object','category']).columns

        ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        train_ohe = ohe.fit_transform(train[cat_feature])
        test_ohe = ohe.transform(test[cat_feature])

        new_cols = (pd.Index(ohe.get_feature_names_out(cat_feature))
                    .str.replace(' ','_')
                    .str.replace('-','_')
                    .str.replace(':','')
                    .str.replace('_/','')
                    .str.replace('/','_')
                    .str.replace('.','_')
                    .str.replace(',','_'))
        print(new_cols.tolist())
        train_ohe_df = pd.DataFrame(train_ohe, columns=new_cols, index=train.index)
        test_ohe_df = pd.DataFrame(test_ohe, columns=new_cols, index=test.index)

        train = train.drop(cat_feature, axis=1).join(train_ohe_df)
        test = test.drop(cat_feature, axis=1).join(test_ohe_df)
    return train, test

# 筛选特征
def feature_selection(train,test,target,fig=False):
    # 划分训练验证集
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42, stratify=target)
    #%%
    # 喂数据
    model = LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=1000,
        verbose=100,
        n_jobs=4,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],  # 验证集
    )
    # 评分
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    print(f"AUC: {auc:.4f}")
    # 特征重要性
    feature_importance = model.feature_importances_
    print('特征重要性：', feature_importance)
    print('feature_importance的数据类型：', feature_importance.dtype)
    #%%
    # 1. 获取特征名
    feature_names = train.columns

    # 2. 创建 DataFrame
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    # 3. 排序
    fi_df = fi_df.sort_values('importance', ascending=False)

    # 4. 打印前 20 名
    print("=== Top 20 最重要特征 ===")
    print(fi_df.head(20))

    # 5. 画图
    if fig:
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), fi_df['importance'].head(20)[::-1])
        plt.yticks(range(20), fi_df['feature'].head(20)[::-1])
        plt.xlabel('Split Count (分裂次数)')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.show()

    # 6. 检查有多少特征重要性小于20
    un_import_count = (fi_df['importance'] < 20).sum()
    print(f"\n共有 {un_import_count} 个特征的重要性小于20 (几乎完全没用)")
    #%%
    # 删除不重要特征
    useful_features = fi_df[fi_df['importance'] > 20]['feature'].tolist()
    train_reduced = train[useful_features]
    test_reduced = test[useful_features]

    print(f"特征从 {train.shape[1]} 个减少到 {len(useful_features)} 个")
    return train_reduced, test_reduced