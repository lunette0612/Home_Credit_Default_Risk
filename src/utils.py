import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


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

# 查看缺失值分布
def nan_distribution(dic):
    y_ticks_train = []
    for col in dic['train'].columns:
        y_ticks_train.append(dic['train'][col].isna().sum() / dic['train'].shape[0] * 100)
    y_ticks_test = []
    for col in dic['test'].columns:
        y_ticks_test.append(dic['test'][col].isna().sum() / dic['test'].shape[0] * 100)
    plt.figure(figsize=(12, 8))
    x = np.arange(len(dic['train'].columns.tolist()))
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