import matplotlib.pyplot as plt # 可视化模块
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
from auto_learning import *
from feature_select import *

##数据读取
dat=pd.read_csv(r'press.csv').iloc[:,1:]
paras_input={'features':dat,
             'date_col':'data_date',##数据中“日期”列名称
             'target_col':'pwh',##数据中“目标变量”列名称
             'fill_strategy':'constant',##constant,drop
             'fill_value':-999,##0,-999等，根据业务定义
             'time_splt':0.8,##0-1百分数
             'test_size':0.3,##0-1百分数
             'scale_flag':'none',##['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler','Normalizer','none']
             'missing_value':np.nan,##数据中缺失值
             'random_state': 21,  ##0~inf随机数
             'select_flag':'model',##['rfe','rfecv','model']
             'thres':5,##正整数，特征选择个数
             'selector':xgb.XGBRegressor,##特征选择算法,xgb,rf,mutual_info_classif,f_classif
             'model':xgb.XGBRegressor,##模型选择算法，目前只支持sklearn
             'model_paras':{'max_depth':range(3,20),'min_child_weight':range(1,50),
       'n_estimators':[100,120,140,160,180,200],
       'gamma':range(10),'reg_alpha':range(20),'reg_lambda':range(20),
       'colsample_bylevel':[i/10 for i in range(2,10)],'subsample':[i/10 for i in range(2,10)]}}
##预处理
fproc=feature_proc(paras_input)
##做归一化,得到数据集
data_sets=fproc.f_data_sets()


#特征选择
fselect=feature_select(data_sets,paras_input)
features_selected=fselect.feature_best()
dt_sets=f_features_selected(data_sets,features_selected)

#模型训练
##hyper_best
hm=hyper_best(dt_sets,paras_input)
best_paras,best_model,best_results=hm.regressor_best(paras_input['model_paras'])
print(best_results)

##optuna_best
op_best=optu_best(dt_sets,paras_input)
best_args,best_clf,best_ks2ndauc=op_best.regress_opt()
print(best_ks2ndauc)





