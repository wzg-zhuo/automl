# -*- coding: utf-8 -*-
import pandas as pd
import  numpy as np
import warnings
import pickle
warnings.filterwarnings(action='ignore')
from sklearn.model_selection import *
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
from sklearn.preprocessing import *


def get_var_type(data,nvar):
    """
    :param data:
    :param nvar:
    :return: 返回变量类型
    """
    categorical_var = [col for col in data.columns if (data[col].dtype == 'object') or (data[col].dtype=='datetime64[ns]')]
    for col in list(data.columns[data.apply(pd.Series.nunique) < nvar]):
        categorical_var.append(col)
    categorical_var = list(set(categorical_var))
    [categorical_var.remove(col) for col in list(categorical_var) if col.find('_WOE') != -1]
    continuous_var = list(set(data.columns) - set(categorical_var))
    [continuous_var.remove(col) for col in list(continuous_var) if col.find('_WOE') != -1]
    return categorical_var,continuous_var

def f_missing_proc(data,target,strategy,missing_value=np.nan,fill_value=0):
    """
    缺失值处理,填充和删除
    :param data: dataframe
    :param missing_value: 缺失值类型如np.nan，''等，若为特殊字符则替换为np.nan
    :param strategy: 处理策略
    :param nvar: 变量类型判断田间
    :param fill_value: 固定填充值
    :return:
    """
    # data=data.replace(missing_value,np.nan)
    data=data[pd.notnull(target)]

    if strategy=='constant':
        try:
            df=data.replace(missing_value,fill_value)
        except:
            df=data.fillna(fill_value)
    elif strategy=='drop':
        df=data.dropna()
    else:
        df=data.copy()
    return df

def f_missring_sets(splt_sets,strategy,missing_value,fill_value):
    train_df = f_missing_proc(splt_sets['train_x'],splt_sets['train_y'],strategy,missing_value,fill_value=fill_value)
    test_df = f_missing_proc(splt_sets['test_x'],splt_sets['test_y'],strategy,missing_value,fill_value=fill_value)
    future_df = f_missing_proc(splt_sets['data_future'],splt_sets['target_future'],strategy,missing_value,fill_value=fill_value)

    if strategy == 'imp':
        ##train
        categorical_var, continuous_var=get_var_type(train_df,1)
        categorical_train=train_df[categorical_var]
        continuous_train=train_df[continuous_var]
        imp = Imputer(missing_values=missing_value, strategy=strategy, axis=0)
        train_continuous = imp.fit_transform(continuous_train)
        train_continuous = pd.DataFrame(train_continuous, columns=continuous_var)
        train_df = pd.concat([categorical_train, train_continuous], axis=1)
        ##test
        categorical_test=test_df[categorical_var]
        continuous_test=test_df[continuous_var]
        test_continuous = imp.transform(continuous_test)
        test_continuous = pd.DataFrame(test_continuous, columns=continuous_var)
        test_df = pd.concat([categorical_test, test_continuous], axis=1)
        ##future
        categorical_future=future_df[categorical_var]
        continuous_future=future_df[continuous_var]
        future_continuous = imp.transform(continuous_future)
        future_continuous = pd.DataFrame(future_continuous, columns=continuous_var)
        future_df = pd.concat([categorical_future, future_continuous], axis=1)


    train_xx = pd.DataFrame(train_df, columns=splt_sets['train_x'].columns)
    test_xx = pd.DataFrame(test_df, columns=splt_sets['test_x'].columns)
    future_xx = pd.DataFrame(future_df, columns=splt_sets['data_future'].columns)

    splt_sets['train_x'] = train_xx
    splt_sets['test_x'] = test_xx
    splt_sets['data_future'] = future_xx
    return splt_sets

def f_future_splt(dat,data_date,threshold,random_state,target='target',test_size=0.3):
    """
    跨时间测试集划分
    :param dat: dataframe
    :param data_date: 观察点对应列,可以是时间，也可以是数值列
    :param thresh:1-跨时间测试集划分比例
    :param target:目标变量对应列
    :return: data_sets,字典，划分的数据集
    """
    dat=dat[pd.notnull(dat[target])]
    if data_date not in dat.columns:
        dat[data_date]=range(len(dat))

    dat = dat.sort_values(data_date)
    splt = round(len(dat) * threshold)
    dat1 = dat.iloc[:splt, :]
    dat_future = dat.iloc[splt:, :]

    target_now = dat1[target]
    data_now = dat1.drop([data_date, target], axis=1)
    train_x, test_x, train_y, test_y = train_test_split(data_now, target_now, test_size=test_size, random_state=random_state)

    # if threshold<1:
    #     target_future = dat_future[target]
    #     data_future = dat_future[data_now.columns]
    #     splt_sets = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y, 'data_future': data_future,
    #                  'target_future': target_future}
    # else:
    #     splt_sets = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}

    target_future = dat_future[target]
    data_future = dat_future[data_now.columns]
    splt_sets = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y, 'data_future': data_future,
                 'target_future': target_future}
    return splt_sets

def f_scale(train_x,trans_x,scale_flag):
    if scale_flag=='StandardScaler':
        scaler=StandardScaler().fit(train_x)
        trans_df=scaler.fit_transform(trans_x)
    elif scale_flag=='MinMaxScaler':
        min_max_scaler = MinMaxScaler().fit(train_x)
        trans_df = min_max_scaler.fit_transform(trans_x)
    elif scale_flag=='MaxAbsScaler':
        max_abs_scaler = MaxAbsScaler().fit(train_x)
        trans_df = max_abs_scaler.fit_transform(trans_x)
    elif scale_flag=='RobustScaler':
        robust_scaler=RobustScaler().fit(train_x)
        trans_df = robust_scaler.fit_transform(trans_x)
    elif scale_flag=='Normalizer':
        normal_scaler=Normalizer().fit(train_x)
        trans_df = normal_scaler.fit_transform(trans_x)
    else:
        trans_df=train_x
    results_df=pd.DataFrame(trans_df,columns=train_x.columns)
    return results_df

def f_scale_sets(data_sets, scale_flag):
    if scale_flag == 'StandardScaler':
        scaler = StandardScaler().fit(data_sets['train_x'])
        try:
            train_df = scaler.fit_transform(data_sets['train_x'])
        except:
            train_df = np.empty((data_sets['train_x'].shape[0],0))
        try:
            test_df = scaler.fit_transform(data_sets['test_x'])
        except:
            test_df = np.empty((data_sets['test_x'].shape[0],0))

        try:
            future_df = scaler.fit_transform(data_sets['data_future'])
        except:
            future_df = np.empty((data_sets['data_future'].shape[0],0))

    elif scale_flag == 'MinMaxScaler':
        min_max_scaler = MinMaxScaler().fit(data_sets['train_x'])
        try:
            train_df = min_max_scaler.fit_transform(data_sets['train_x'])
        except:
            train_df = np.empty((data_sets['train_x'].shape[0],0))
        try:
            test_df = min_max_scaler.fit_transform(data_sets['test_x'])
        except:
            test_df = np.empty((data_sets['test_x'].shape[0],0))
        try:
            future_df = min_max_scaler.fit_transform(data_sets['data_future'])
        except:
            future_df = np.empty((data_sets['data_future'].shape[0],0))

    elif scale_flag == 'MaxAbsScaler':
        max_abs_scaler = MaxAbsScaler().fit(data_sets['train_x'])
        try:
            train_df = max_abs_scaler.fit_transform(data_sets['train_x'])
        except:
            train_df = np.empty((data_sets['train_x'].shape[0],0))
        try:
            test_df = max_abs_scaler.fit_transform(data_sets['test_x'])
        except:
            test_df = np.empty((data_sets['test_x'].shape[0],0))
        try:
            future_df = max_abs_scaler.fit_transform(data_sets['data_future'])
        except:
            future_df = np.empty((data_sets['data_future'].shape[0],0))
    elif scale_flag == 'RobustScaler':
        robust_scaler = RobustScaler().fit(data_sets['train_x'])
        try:
            train_df = robust_scaler.fit_transform(data_sets['train_x'])
        except:
            train_df = np.empty((data_sets['train_x'].shape[0],0))
        try:
            test_df = robust_scaler.fit_transform(data_sets['test_x'])
        except:
            test_df = np.empty((data_sets['test_x'].shape[0],0))
        try:
            future_df = robust_scaler.fit_transform(data_sets['data_future'])
        except:
            future_df = np.empty((data_sets['data_future'].shape[0],0))
    elif scale_flag=='Normalizer':
        normal_scaler = Normalizer().fit(data_sets['train_x'])
        try:
            train_df = normal_scaler.fit_transform(data_sets['train_x'])
        except:
            train_df = np.empty((data_sets['train_x'].shape[0],0))
        try:
            test_df = normal_scaler.fit_transform(data_sets['test_x'])
        except:
            test_df = np.empty((data_sets['test_x'].shape[0],0))
        try:
            future_df = normal_scaler.fit_transform(data_sets['data_future'])
        except:
            future_df = np.empty((data_sets['data_future'].shape[0],0))
    else:
        try:
            train_df = data_sets['train_x']
        except:
            train_df = np.empty((data_sets['train_x'].shape[0],0))
        try:
            test_df = data_sets['test_x']
        except:
            test_df = np.empty((data_sets['test_x'].shape[0],0))
        try:
            future_df = data_sets['data_future']
        except:
            future_df = np.empty((data_sets['data_future'].shape[0],0))
    try:
        train_xx = pd.DataFrame(train_df, columns=data_sets['train_x'].columns)
    except:
        train_xx = pd.DataFrame(columns=data_sets['train_x'].columns)
    try:
        test_xx = pd.DataFrame(test_df, columns=data_sets['test_x'].columns)
    except:
        test_xx = pd.DataFrame(columns=data_sets['test_x'].columns)
    try:
        future_xx = pd.DataFrame(future_df, columns=data_sets['data_future'].columns)
    except:
        future_xx = pd.DataFrame(columns=data_sets['data_future'].columns)
    data_sets['train_x'] = train_xx
    data_sets['test_x'] = test_xx
    data_sets['data_future'] = future_xx

    return data_sets


class feature_proc:
    def __init__(self,paras_input):
        self.features=paras_input['features']
        self.data_date=paras_input['date_col']
        self.target=paras_input['target_col']
        self.random_state=paras_input['random_state']
        self.fill_strategy=paras_input['fill_strategy']
        self.fill_value=paras_input['fill_value']
        self.time_splt=paras_input['time_splt']
        self.test_size=paras_input['test_size']
        self.scale_flag=paras_input['scale_flag']
        self.missing_value=paras_input['missing_value']

    

    def f_data_sets(self):
        split_sets = f_future_splt(self.features, self.data_date, self.time_splt, self.random_state, self.target, self.test_size)
        proc_sets = f_missring_sets(split_sets,self.fill_strategy,self.missing_value,self.fill_value)
        data_sets = f_scale_sets(proc_sets, self.scale_flag)
        train_dim=data_sets['train_x'].shape
        train_r=data_sets['train_y'].sum()/train_dim[0]
        test_dim=data_sets['test_x'].shape
        test_r=data_sets['test_y'].sum()/test_dim[0]
        if self.time_splt<1:
            future_dim=data_sets['data_future'].shape
            future_r=data_sets['target_future'].sum()/future_dim[0]
            print('train_dim:', train_dim, 'test_dim:', test_dim, 'future_dim:', future_dim)
            print('train_r:', train_r, 'test_r:', test_r, 'future_r:', future_r)
        else:
            print('train_dim:', train_dim, 'test_dim:', test_dim)
            print('train_r:', train_r, 'test_r:', test_r)

        return data_sets


    #
    # def f_setscaler(self):
    #     data_sets=self.f_future_splt()
    #     dt_sets = f_scale_sets(data_sets, self.scale_flag)
    #     train_dim=dt_sets['train_x'].shape
    #     test_dim=dt_sets['test_x'].shape
    #     future_dim=dt_sets['data_future'].shape
    #     train_r=dt_sets['train_y'].sum()/train_dim[0]
    #     test_r=dt_sets['test_y'].sum()/test_dim[0]
    #     future_r=dt_sets['target_future'].sum()/future_dim[0]
    #
    #     print('train_dim:', train_dim, 'test_di:', test_dim, 'future_dim:', future_dim)
    #     print('train_r:', train_r, 'test_r:', test_r, 'future_r:', future_r)
    #
    #     return dt_sets
    #
    # def f_scaler_sets(self):
    #     dat_proc = f_missing_proc(self.features, self.target, self.fill_strategy, fill_value=self.fill_value)
    #     dt_tmp = f_future_splt(dat_proc, self.data_date, self.threshold, self.random_state, self.target, self.test_size)
    #     dt_sets = f_scale_sets(dt_tmp, self.scale_flag)
    #     return dt_sets


if __name__ == '__main__':
    data = pd.read_csv(r'features.csv')
    data=f_missing_proc(data,'target','c')
    data_sets = f_future_splt(data.iloc[:,1:], 'data_date', 0.2, 1000, 'target',0.3)

