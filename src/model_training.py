from lightgbm import LGBMClassifier
from sklearn.model_selection import  GridSearchCV,StratifiedKFold,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.stats import uniform, randint


# 以LGBM作为主模型，对它进行调参
def lgbm_HPO(X,y):
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
    random_gridsearch.fit(X, y)
    best_params = random_gridsearch.best_params_
    print(f'最好的参数：{best_params}')
    print(f'最好的分数：{random_gridsearch.best_score_}')

# knn和svm先调个参数
def knn_svc_HPO(X,y):
    knn_grid = GridSearchCV(
    Pipeline([
        ('scaler',StandardScaler()),
        ('knn', KNeighborsClassifier())
    ]),
        param_grid={'knn__n_neighbors': list(range(2,10))},
        scoring='roc_auc',
        cv=3,
    )
    knn_grid.fit(X, y)
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
    svm_grid.fit(X, y)
    best_svm_params = svm_grid.best_params_
    print(f'svm最好参数{best_svm_params}')

    # 去除 'svm__' 前缀
    clean_svm_params = {key.split('__')[1]: value for key, value in best_svm_params.items()}
    return clean_knn_params,clean_svm_params

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

# 模型集成
def model_ensemble(X,y,new_main_model,knn_pipeline,lr_pipeline,evaluate=True,svm_pipeline=None):
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
        cv_scores = cross_val_score(stacking, X, y, cv=3, scoring='roc_auc', n_jobs=4)
        print(f"Stacking CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print('开始训练！')
    stacking.fit(X, y)
    print('训练完成！')

    # 查看权重
    coefs = stacking.final_estimator_.named_steps['lr'].coef_[0]
    print(f"\n元学习器系数（基学习器权重）:")
    for i, name in enumerate(['XGB', 'LGBM', 'RF','GAUSSIAN','KNN']):
        print(f"{name}:{coefs[i]:.4f}")
    return stacking