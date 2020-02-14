import numpy as np
import pandas as pd
from feature_select import *
from scipy import stats
import xgboost as xgb


def f_feature_corr(feature_data,cols_imp,corr_thres,var_del):
    """

    :param feature_data: 变量df
    :param cols_imp: 变量重要性信息，包括变量名“var”和特征重要性imp两列
    :param varname: cols_imp中变量名所在列名称
    :param imp: cols_imp中imp值所在列名称
    :param corr_thres: 相关性阈值
    :param var_del: list调整后需要删除变量，根据初步结果查看相关性删除变量特征，后人工决策剔除部分变量
    :return: feature_del_lst根据相关性需要剔除的变量list
    """

    feature_df=feature_data.drop(var_del,axis=1)
    cols_imp=cols_imp[~cols_imp['var'].isin(var_del)]
    con_woe2ndiv_inf = cols_imp[['var', 'imp']].drop_duplicates().sort_values(by=['imp'], ascending=False)
    con_woe2ndiv_asc = cols_imp[['var', 'imp']].drop_duplicates().sort_values(by=['imp'], ascending=True)

    vars_list = list(con_woe2ndiv_inf['var'])
    vars_asc = list(con_woe2ndiv_asc['var'])
    len(vars_list)
    feature_del = []
    feature_del_df=pd.DataFrame()
    for var_imp_min in vars_asc[:-1]:
        for i in range(0, len(vars_list) - 1):
            var_imp_high = vars_list[i]
            corr, pval = stats.spearmanr(feature_df[var_imp_min], feature_df[var_imp_high])
            print(corr)
            if np.abs(corr) >= corr_thres:
                feature_del.append(var_imp_min)
                feature_del_temp=pd.DataFrame({'var_imp_high':var_imp_high,'corr':corr},index=[var_imp_min])
                feature_del_df=pd.concat([feature_del_df,feature_del_temp])
        vars_list.remove(var_imp_min)
    feature_del_lst=list(set(feature_del))
    feature_del_lst.sort(key=feature_del.index)
    feature_del_df.reset_index(inplace=True)
    feature_selected = [col for col in feature_df.columns if col not in feature_del_lst]

    return feature_del_lst,feature_del_df,feature_selected

if __name__=='main':
    data = pd.read_csv(r'features.csv')
    data=data.iloc[:,1:]
    data=f_missing_proc(data,np.nan,'c',1,0)
    cols_imp=f_model_rank(xgb.XGBClassifier(),data.drop(['target','UUID','data_date'],axis=1),data.target)
    feature_deleted_lst,feature_deleted_df,feature_selected=f_feature_corr(data,cols_imp,0.7,['UUID','data_date','target'])
