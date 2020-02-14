# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore')
from feature_corr_selected import *
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
from scipy.stats import pearsonr
from sklearn.feature_selection import *
from sklearn.ensemble import *

def f_pearson(train_x,train_y):
    return np.array([pearsonr(train_x[col],train_y)[0] for col in train_x.columns])

def f_chi2(train_x,train_y):
    return chi2(train_x,train_y)[0]

def f_feature_best(selector,thres,train_x,train_y,random_state,flag=None):
    """
    :param selector:
    :param thres:
    :param train_x:
    :param train_y:
    :param flag:
    :return:
    """
    if flag in ['rfe','rfecv','model']:
        cols_rank = f_model_rank(selector, train_x, train_y,random_state, flag=flag).sort_values('imp',ascending=False)
    else:
        cols_rank = f_desc_rank(selector, train_x, train_y).sort_values('imp',ascending=False)
    select_cols=cols_rank.iloc[:thres,:]['var']
    cols_select = select_cols.tolist()

    return cols_select


def f_desc_rank(func,train_x,train_y):
    try:
        pr=func(train_x,train_y).tolist()
    except:
        pr=func(train_x,train_y)[0].tolist()

    cols_rank=pd.DataFrame(dict(zip(train_x.columns.tolist(),pr)),index=[1]).T.reset_index().fillna(0)

    cols_rank.columns=['var','imp']
    return cols_rank

def f_model_rank(model,train_x,train_y,random_state=21,flag=None):
    if flag=='rfe':
        selector_rank=RFE(model(random_state=random_state),5,step=1).fit(train_x,train_y).ranking_
    elif flag=='rfecv':
        selector_rank=RFECV(model(random_state=random_state),5).fit(train_x,train_y).ranking_
    else:
        selector_rank=model(random_state=random_state).fit(train_x,train_y).feature_importances_
    cols_rank=pd.DataFrame(dict(zip(train_x.columns.tolist(),selector_rank.tolist())),index=[1]).T.reset_index()
    cols_rank.columns=['var','imp']
    return cols_rank

def f_features_selected(data_sets,features_selected):
    for dk,dv in data_sets.items():
        try:
            data_sets[dk]=dv[features_selected]
        except:
            data_sets[dk]=dv
    return data_sets


class feature_select:
    def __init__(self,proc_sets,paras_input):
        self.data_sets=proc_sets
        self.select_flag=paras_input['select_flag']
        self.random_state=paras_input['random_state']
        self.thres=paras_input['thres']
        self.selector=paras_input['selector']

    def feature_best(self):
        cols_select=f_feature_best(self.selector,self.thres,self.data_sets['train_x'],self.data_sets['train_y'],self.random_state,self.select_flag)
        return cols_select

    def desc_rank(self,func):
        cols_rank=f_desc_rank(func,self.data_sets['train_x'],self.data_sets['train_y'])
        return cols_rank

    def model_rank(self):
        cols_rank=f_model_rank(self.selector,self.data_sets['train_x'],self.data_sets['train_y'],self.select_flag)
        return cols_rank

    def select_feature(self,features_selected):
        select_sets=f_features_selected(self.data_sets,features_selected)
        return select_sets



