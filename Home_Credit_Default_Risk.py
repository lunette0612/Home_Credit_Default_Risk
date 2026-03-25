import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import gc
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold,cross_val_score,RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.stats import uniform, randint

#%%
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # 对于对象类型，如果唯一值较少，转为 category 可以极大节省内存
            if df[col].dtype == 'object':
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:  # 阈值可调整
                    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print(
        f'Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def read_data():
    train = reduce_mem_usage(pd.read_csv('application_train.csv'))
    test = reduce_mem_usage(pd.read_csv('application_test.csv'))
    previous = reduce_mem_usage(pd.read_csv('previous_application.csv'))
    POS_CASH_balance = reduce_mem_usage(pd.read_csv('POS_CASH_balance.csv'))
    installments = reduce_mem_usage(pd.read_csv('installments_payments.csv'))
    credit_card_balance = reduce_mem_usage(pd.read_csv('credit_card_balance.csv'))
    bureau_balance = reduce_mem_usage(pd.read_csv('bureau_balance.csv'))
    bureau = reduce_mem_usage(pd.read_csv('bureau.csv'))
    return {
        'train': train,
        'test': test,
        'previous': previous,
        'POS_CASH_balance': POS_CASH_balance,
        'installments': installments,
        'credit_card_balance': credit_card_balance,
        'bureau_balance': bureau_balance,
        'bureau': bureau
    }

print('正在读取数据...')
data = read_data()

#%%
# 分离出训练集标签
target = data['train']['TARGET']
data['train'].drop('TARGET', axis=1, inplace=True)

# 查看缺失值分布
def nan_distribution():
    y_ticks_train = []
    for col in data['train'].columns:
        y_ticks_train.append(data['train'][col].isna().sum() / data['train'].shape[0] * 100)
    y_ticks_test = []
    for col in data['test'].columns:
        y_ticks_test.append(data['test'][col].isna().sum() / data['test'].shape[0] * 100)
    plt.figure(figsize=(12, 8))
    x = np.arange(len(data['train'].columns.tolist()))
    width = 0.35
    plt.bar(
        x - width / 2,
        y_ticks_train,
        width,
        color='red'
    )
    plt.bar(
        x + width / 2,
        y_ticks_test,
        width,
        color='blue'
    )
    plt.show()
nan_distribution()

#%%
# 连接所有表
def table_merge():
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
        for col in data['bureau_balance'].columns:
            if col == 'SK_ID_CURR' or col == 'SK_ID_BUREAU':
                continue
            elif data['bureau_balance'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_bureau_balance = data['bureau_balance'].groupby('SK_ID_BUREAU').agg(agg_dict)
        # 扁平化列名
        agg_bureau_balance.columns = ['bureau_balance_' + '_'.join(col).upper() for col in agg_bureau_balance.columns]
        # 重置索引
        agg_bureau_balance = agg_bureau_balance.reset_index()
        return agg_bureau_balance

    print('正在合并bureau_balance')
    agg_bureau_balance = bureau_balance_agg()
    bureau = pd.merge(data['bureau'], agg_bureau_balance, how='left', on='SK_ID_BUREAU')
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
    train_temp = pd.merge(data['train'], agg_bureau, how='left', on='SK_ID_CURR')
    test_temp = pd.merge(data['test'], agg_bureau, how='left', on='SK_ID_CURR')

    # 聚合POS_CASH_balance表
    def POS_CASH_balance_agg():
        agg_dict = {}
        for col in data['POS_CASH_balance'].columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue
            elif data['POS_CASH_balance'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_POS_CASH_balance = data['POS_CASH_balance'].groupby('SK_ID_PREV').agg(agg_dict)
        agg_POS_CASH_balance.columns = ['POS_CASH_balance_' + '_'.join(col).upper() for col in
                                        agg_POS_CASH_balance.columns]
        agg_POS_CASH_balance = agg_POS_CASH_balance.reset_index()
        return agg_POS_CASH_balance

    agg_POS_CASH_balance = POS_CASH_balance_agg()
    print('正在合并previous0')
    previous0 = pd.merge(data['previous'], agg_POS_CASH_balance, how='left', on='SK_ID_PREV')
    del agg_POS_CASH_balance

    # 聚合installments表
    def installments_agg():
        agg_dict = {}
        for col in data['installments'].columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue
            elif data['installments'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_installments = data['installments'].groupby('SK_ID_PREV').agg(agg_dict)
        # 计算每次分期的逾期天数（应还日期 - 实际还款日期）
        # 负数表示逾期，正数表示提前还款
        data['installments']['OVERDUE_DAYS'] = data['installments']['DAYS_INSTALMENT'] - data['installments'][
            'DAYS_ENTRY_PAYMENT']

        # 计算每次分期的还款比例
        data['installments']['PAYMENT_RATIO'] = data['installments']['AMT_PAYMENT'] / data['installments'][
            'AMT_INSTALMENT'].replace(0, np.nan)

        # 标记是否逾期（逾期天数 > 0 表示逾期）
        data['installments']['IS_OVERDUE'] = (data['installments']['OVERDUE_DAYS'] > 0).astype(int)

        # 对新增字段做聚合
        overdue_agg = data['installments'].groupby('SK_ID_PREV').agg({
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
        for col in data['credit_card_balance'].columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue
            elif data['credit_card_balance'][col].dtype == np.number:  # 数值列
                agg_dict[col] = ['min', 'max', 'mean', 'sum', 'std',
                                 ('p25', lambda x: x.quantile(0.25)),
                                 ('p75', lambda x: x.quantile(0.75))]
            else:  # 类别列
                agg_dict[col] = ['count', 'nunique']
        agg_credit_card_balance = data['credit_card_balance'].groupby('SK_ID_PREV').agg(agg_dict)
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
    del data['bureau'], data['bureau_balance'], data['POS_CASH_balance']
    del data['installments'], data['credit_card_balance'], data['previous']
    gc.collect()

    return train, test

train, test = table_merge()

#%%
# 分离出测试集ID
curr_id = test['SK_ID_CURR']
test.drop(['SK_ID_CURR'], axis=1, inplace=True)
train.drop(['SK_ID_CURR'], axis=1, inplace=True)
# train.info(verbose=True)

#%%
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

# 应用填充函数
train, test = fill(train, test)

#%%
# 验证是否填充完毕
def verify(df):
    aa = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            aa.append(col)
    if len(aa) > 0:
        print(f"未填充完毕！未填充列表:{aa}，长度{len(aa)}")
    else:
        print("填充完毕！")
verify(train)
verify(test)

#%%
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
        df['STABILITY_SCORE'] = (
            abs(df['DAYS_EMPLOYED']) +
            abs(df['DAYS_REGISTRATION']) +
            abs(df['DAYS_LAST_PHONE_CHANGE'])
        ) / 3

        # 债务负担综合评分
        df['DEBT_BURDEN_SCORE'] = (
            df['CREDIT_INCOME_RATIO'] +
            df['ANNUITY_INCOME_RATIO'] +
            df['CREDIT_ANNUITY_RATIO']
        ) / 3

        # 比率极值
        df['RATIO_MAX'] = df[[
            'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
            'CREDIT_ANNUITY_RATIO', 'AMT_ANNUITY_GOODS_RATIO'
        ]].max(axis=1)

        df['RATIO_MIN'] = df[[
        'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
        'CREDIT_ANNUITY_RATIO', 'AMT_ANNUITY_GOODS_RATIO'
        ]].min(axis=1)

        # 每孩子贷款
        df['CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / (df['CNT_CHILDREN'] + 1)

        # 金额比率
        df['ANNUITY_PER_PERSON'] = df['AMT_ANNUITY'] / (df['INCOME_PER_PERSON'] + 1)

        # 信贷密度
        df['CREDIT_DENSITY'] = df['CREDIT_PER_CHILD'] * df['REGION_POPULATION_RELATIVE']

        # 高债务 + 低稳定性 = 高风险
        df['HIGH_RISK_SCORE'] = df['DEBT_BURDEN_SCORE'] * (1 / (df['STABILITY_SCORE'] + 1))

        # 外部评分 + 债务负担
        df['EXT_DEBT_INTERACTION'] = df['EXT_SOURCE_MEAN'] * df['DEBT_BURDEN_SCORE']

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
        income_labels = ['INC_0_0.125', 'INC_0.125_0.25', 'INC_0.25_0.375', 'INC_0.375_0.5','INC_0.5_0.625', 'INC_0.625_0.75', 'INC_0.75_0.875', 'INC_0.875_1']

        df['INCOME_BIN'] = pd.cut(df['AMT_INCOME_TOTAL'],
                          bins=income_bins,
                          labels=income_labels,
                          right=True,
                          include_lowest=True)

        # 转为二值特征
        income_dummies = pd.get_dummies(df['INCOME_BIN'], prefix='INCOME')
        for col in income_dummies.columns:
            df[col] = income_dummies[col].astype(int)

        # 是否有孩子 × 收入等级
        df['IS_HIGH_INCOME_WITH_CHILD'] = ((df['AMT_INCOME_TOTAL'] > 50000) & (df['CNT_CHILDREN'] > 0)).astype(int)

        # 工作年限/年龄比
        df['EMPLOYED_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

        # 年龄（年）
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

        # 年龄异常
        df['AGE_OUTLIER'] = ((df['AGE_YEARS'] < 18) | (df['AGE_YEARS'] > 70)).astype(int)

        # 年龄分箱
        bins = [0, 20, 30, 40, 50, 60, 200]
        labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-200']
        df['AGE_BIN'] = pd.cut(df['AGE_YEARS'], bins=bins, labels=labels, right=True,include_lowest=True)
        dummies = pd.get_dummies(df['AGE_BIN'], prefix='AGE')
        for col in dummies.columns:
            df[col] = dummies[col].astype(int)

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

        # 加和计数（表示资产丰富程度）
        df['ASSET_COUNT'] = df['HAS_CAR'] + df['HAS_REALTY'] + df['HAS_CHILD']

    return train, test
train, test = add_features(train, test)

#%%
# 再次填充
train, test = fill(train, test)
verify(train)
verify(test)

#%%
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

train, test = cat_feature_encoder(train, test,encoder='onehot')

#%%
# 再次挤压内存
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

#%%
# 筛选特征
def feature_selection():
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
train_reduced, test_reduced = feature_selection()

#%%
# 以LGBM作为主模型，对它进行调参
def lgbm_HPO():
    main_model = LGBMClassifier(class_weight='balanced',n_estimators=1000,metric='roc_auc')

    param_distributions = {
        'num_leaves': randint(30,100),
        'max_depth': randint(3,9),
        'reg_alpha': uniform(0,1),
        'reg_lambda': uniform(0,1),
        'learning_rate': uniform(0.01,0.2),
    }
    # 网格随机搜索
    random_gridsearch = RandomizedSearchCV(
        estimator=main_model,
        param_distributions=param_distributions,
        scoring='roc_auc',
        n_iter=40,
        random_state=42,
        n_jobs=4)
    random_gridsearch.fit(train_reduced, target)
    best_params = random_gridsearch.best_params_
    print(f'最好的参数：{best_params}')
    print(f'最好的分数：{random_gridsearch.best_score_}')
# lgbm_HPO()
# 最好的参数：{'learning_rate': np.float64(0.08169314570885453), 'max_depth': 3, 'num_leaves': 57, 'reg_alpha': np.float64(0.8631034258755935), 'reg_lambda': np.float64(0.6232981268275579)}
# 最好的分数：0.7743393515494559

#%%
# 不能将见过训练数据的model传入集成模型，只能拿到调好的参数重新实例化一个模型
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

#%%
# knn和svm先调个参数
def knn_svc_HPO():
    knn_grid = GridSearchCV(
    Pipeline([
        ('scaler',StandardScaler()),
        ('knn', KNeighborsClassifier())
    ]),
        param_grid={'knn__n_neighbors': list(range(2,10))},
        scoring='roc_auc',
        cv=3,
    )
    knn_grid.fit(train_reduced, target)
    best_knn_params = knn_grid.best_params_
    print(f'knn最好参数：{best_knn_params}')

    # 去除 'knn__' 前缀
    #  best_knn_params 格式：{'knn__n_neighbors': 5}
    #  需要变成：{'n_neighbors': 5}
    clean_knn_params = {key.split('__')[1]: value for key, value in best_knn_params.items()}

    # svm同理
    svm_grid = GridSearchCV(
        Pipeline([
        ('scaler',StandardScaler()),
        ('svm', SVC(probability=True,kernel='rbf', random_state=42))
        ]),
        param_grid={'svm__C': [0.1, 1, 10]},
        scoring='roc_auc',
        cv=3,
    )
    svm_grid.fit(train_reduced, target)
    best_svm_params = svm_grid.best_params_
    print(f'svm最好参数{best_svm_params}')

    # 去除 'svm__' 前缀
    clean_svm_params = {key.split('__')[1]: value for key, value in best_svm_params.items()}
    return clean_knn_params,clean_svm_params
# clean_knn_params,clean_svm_params = knn_svc_HPO()

# 拿到调好的模型放到管道
def model_pipeline():
    knn_pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=9,n_jobs=4)),
    ])
    svm_pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('svm', SVC(probability=True,kernel='rbf', random_state=42,C=1))
    ])
    # LR管道
    lr_pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('lr', LogisticRegression(random_state=42,max_iter=3000,n_jobs=4))
    ])
    return knn_pipeline, svm_pipeline, lr_pipeline
knn_pipeline, svm_pipeline, lr_pipeline = model_pipeline()

#%%
# 模型集成
def model_ensemble(evaluate=True):
    """
    基学习器：XGB,LGBM,RF，GAUSSIAN,KNN,SVM
    元学习器：LR
    :return:stacking
    """
    stacking = StackingClassifier(
        estimators=[
            ('XGB', XGBClassifier(n_jobs=4)),
            ('LGBM', new_main_model),
            ('RF', RandomForestClassifier(n_jobs=4)),
            ('GAUSSIAN', GaussianNB()),
            ('KNN',knn_pipeline),
            # ('SVM', svm_pipeline)
        ],
        final_estimator=lr_pipeline,
        cv=StratifiedKFold(n_splits=2,shuffle=True, random_state=42),
        n_jobs=4,
        passthrough=False,
    )

    # 交叉评估
    if evaluate:
        cv_scores = cross_val_score(stacking, train_reduced, target, cv=3, scoring='roc_auc', n_jobs=4)
        print(f"Stacking CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print('开始训练！')
    stacking.fit(train_reduced, target)
    print('训练完成！')

    # 查看权重
    coefs = stacking.final_estimator_.named_steps['lr'].coef_[0]
    print(f"\n元学习器系数（基学习器权重）:")
    for i, name in enumerate(['XGB', 'LGBM', 'RF','GAUSSIAN','KNN']):
        print(f"{name}:{coefs[i]:.4f}")
    return stacking
stacking = model_ensemble(evaluate=False)


# 集成模型和LGBM分别预测拿到结果
if __name__ == '__main__':
    # 集成模型预测
    proba = stacking.predict_proba(test_reduced)[:, 1]
    pd.DataFrame({
        'SK_ID_CURR': curr_id,
        'TARGET': proba
    }).to_csv('sub_stacking.csv', index=False)
    #%%
    # 只用LGBM预测
    new_main_model.fit(train_reduced, target)
    proba = new_main_model.predict_proba(test_reduced)[:, 1]
    pd.DataFrame({
        'SK_ID_CURR': curr_id,
        'TARGET': proba
    }).to_csv('sub_LGB.csv', index=False)#