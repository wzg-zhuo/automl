import pandas as pd
import warnings
from sklearn.metrics import *
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
from feature_proc import *
from sklearn.metrics import *

def ks_results(x,target,model):
    prob_target=pd.DataFrame(target)
    prob_target['prob'] = model.predict_proba(x)[:, 1]
    cross1 = pd.crosstab(prob_target['prob'], prob_target['target'])
    cross1 = cross1.reset_index()
    if 0 not in cross1.columns:
        cross1[0] = 0
    if 1 not in cross1.columns:
        cross1[1] = 0
    cross1['y_cum_r'] = cross1[1].cumsum() / cross1[1].sum()
    cross1['n_cum_r'] = cross1[0].cumsum() / cross1[0].sum()
    cross1['ks'] = cross1['n_cum_r'] - cross1['y_cum_r']
    KS = cross1['ks'].abs().max()
    return KS

def auc_results(x, target, model):
    prob = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(target, prob)
    return auc

def regress_results(data, target, model):
    predict = model.predict(data)
    # 平均绝对值误差（mean_absolute_error）
    mae = mean_absolute_error(target, predict)
    # 决定系数（R-Square）
    r2 = r2_score(target, predict)
    # 校正决定系数（Adjusted R-Square）
    r2_adj = 1 - (1 - r2) * (len(data) - 1) / (len(data) - data.shape[1] - 1)
    # 均方误差 MSE（Mean Squared Error）
    mse = mean_squared_error(target, predict)

    # #均方根对数误差 RMSLE（Root Mean Squared Logarithmic Error）
    # train_rmse=mean_squared_log_error(train_target,train_predict)
    # test_rmse=mean_squared_log_error(test_target,test_predict)
    # 平均绝对误差 MAD(mean absolute deviation)
    def MAD(target, predictions):
        absolute_deviation = np.abs(target - predictions)
        return np.mean(absolute_deviation)

    msd = MAD(target, predict)

    # 平均绝对百分比误差（Mean Absolute Percentage Error）
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / (y_true + 10 ** (-5)))) * 100

    mape = mape(target, predict)

    # 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
    def smape(y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

    smape = smape(target, predict)
    # 解释方差(explained_variance_score)
    evs = explained_variance_score(target, predict)

    # 标准残差图
    def std_e(y_true, y_pred, train_data):
        e = y_true - y_pred
        sse = np.sum(np.square(e))
        se = sse / (len(y_true) - len(train_data) - 1)
        try:
            d = e / se
        except:
            d = 0
        return d

    e = std_e(target, predict, data)
    e_df = pd.DataFrame(e).reset_index()
    e_df.columns = ['x_col', 'e']

    name_lst = [' R-Square',
                'Adjusted R-Square',
                'explained_variance_score',
                'mean_absolute_error',
                # '均方根对数误差 RMSLE（Root Mean Squared Logarithmic Error）',
                'mean absolute deviation',
                'Mean Absolute Percentage Error',
                'Symmetric Mean Absolute Percentage Error',
                'Mean Squared Error']

    bz_lst = [' 拟合优度',
              '校正决定系数',
              '解释方差',
              '平均绝对值误差',
              # '均方根对数误差',
              '平均绝对误差',
              '平均绝对百分比误差',
              '对称平均绝对百分比误差',
              '均方误差']

    vl = [r2, r2_adj, evs, mae, msd, mape, smape, mse]

    train_dct = dict(zip(name_lst, vl))

    model_test = pd.DataFrame(train_dct, index=[0]).T.reset_index()
    model_test.columns = ['评估指标', '评估指标值']
    model_test['备注'] = bz_lst

    predict = pd.DataFrame(predict, index=range(len(predict)))
    target.index = range(len(target))
    results = pd.concat([target, predict], axis=1)
    results.columns = ['real', 'predict']

    return model_test, results,e_df

def f_regress_results(rf_model,data_sets):
    train_results,train_predict_df,train_e_df=regress_results(data_sets['train_x'],data_sets['train_y'],rf_model)
    test_results,test_predict_df,test_e_df=regress_results(data_sets['test_x'],data_sets['test_y'],rf_model)
    future_results,future_predict_df,future_e_df=regress_results(data_sets['data_future'],data_sets['target_future'],rf_model)

    results=pd.merge(pd.merge(train_results,test_results,on=['评估指标','备注']),future_results,on=['评估指标','备注'])
    results.columns=['评估指标','评估指标值-训练集','备注','评估指标值-验证集','评估指标集-测试集']
    results=results[['评估指标','评估指标值-训练集','评估指标值-验证集','评估指标集-测试集','备注']]

    return results

def f_features_ana(data,target,vars_name,flag,model=None):
    vars=data.drop([target],axis=1).columns.tolist()
    vn_dct=dict(zip(vars,vars_name))
    if flag=='model':
        model.fit(data.drop([target],axis=1), data[target])
        imp=model.feature_importances_
        vars_imp=pd.DataFrame(dict(zip(vars,imp)),index=[0]).T.reset_index()
        vars_imp.columns=['变量名','特征重要性']
        vars_imp.变量名=vars_imp.变量名.replace(vn_dct)
    elif flag=='mr':
        mr_inf=dict()
        for v in data.columns:
            mr_tmp=mutual_info_score(data[v],data[target])
            mr_inf[v]=mr_tmp
        vars_imp=pd.DataFrame(mr_inf,index=[0]).T.reset_index()
        vars_imp.columns=['变量名','特征重要性']
        vars_imp.sort_values('特征重要性',ascending=False)
        vars_imp.变量名=vars_imp.变量名.replace(vn_dct)
    vars_imp=vars_imp.sort_values('特征重要性',ascending=False)

    return vars_imp
