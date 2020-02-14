# -*- coding: utf-8 -*-
import pandas as pd
import  numpy as np
import sys,os
import pickle
import warnings
warnings.filterwarnings(action='ignore')
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from functools import reduce
from datetime import timedelta
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

##数据预处理,根据实际情况而定
def data_process(air_df0, data_date, depdate, t0):
    ##onehot操作
    def f_tbins(x):
        if x > t1 and x <= t2:
            return '7_12'
        elif x > t2 and x <= t3:
            return '12_17'
        elif x > t3 and x <= t4:
            return '17_22'
        else:
            return '22_7'

    def f_clsbins(x):
        if x == 'Y':
            return 'Y'
        elif x == 'C':
            return 'C'
        elif x == 'F':
            return 'F'
        else:
            return 'OT'

    def f_level_bins(x):
        if x > 0:
            return 'Hi'
        elif x < 0:
            return 'Lw'
        else:
            return 'Mid'

    def date_process(air_df):
        air_df['data_date'] = pd.to_datetime ( air_df['data_date'], errors='coerce', format='%Y-%m-%d' )
        air_df['depdate'] = pd.to_datetime ( air_df['depdate'], errors='coerce', format='%Y-%m-%d' )
        air_df['born_dt'] = pd.to_datetime ( air_df['born_dt'], errors='coerce', format='%H-%m' )
        air_df['seg_arrv_dt_lcl'] = pd.to_datetime ( air_df['seg_arrv_dt_lcl'], errors='coerce', format='%Y-%m-%d' )
        air_df['tkt_bkg_dt'] = pd.to_datetime ( air_df['tkt_bkg_dt'], errors='coerce', format='%Y-%m-%d' )
        air_df['seg_arrv_tm_lcl'] = pd.to_datetime ( air_df['seg_arrv_tm_lcl'], format='%H:%M:%S' )
        air_df['seg_dpt_tm_lcl'] = pd.to_datetime ( air_df['seg_dpt_tm_lcl'], format='%H:%M:%S' )
        return air_df

    air_df1 = date_process ( air_df0 )

    def numeric_proces(air_df):
        air_df = air_df.replace ( "\\N", np.nan )
        air_df['discount'] = pd.to_numeric ( air_df['discount'] )
        air_df['discount'] = air_df['discount'].replace ( 0, np.nan )
        air_df['bag_pcs'] = pd.to_numeric ( air_df['bag_pcs'] )
        air_df['tkt_amt'] = pd.to_numeric ( air_df['tkt_amt'] )
        air_df['chkd_bag_wgt_qty'] = pd.to_numeric ( air_df['chkd_bag_wgt_qty'] )
        air_df['chkd_bag_wgt_qty'] = air_df['chkd_bag_wgt_qty'].replace ( 0, np.nan )
        air_df['bag_pcs'] = pd.to_numeric ( air_df['bag_pcs'] )
        air_df['mileage'] = pd.to_numeric ( air_df['mileage'] )
        air_df['yprice'] = pd.to_numeric ( air_df['yprice'] )
        air_df['segprice'] = pd.to_numeric ( air_df['segprice'] )

        return air_df

    air_df2 = numeric_proces ( air_df1 )

    def prefeature_proces(air_df):
        '''
        对节假日类型，旅游机场，等级机场特征计算做预处理
        :param air_df:
        :return:
        '''
        file1 = open ( r'C:\Users\Wangzhuo\Desktop\air2ndrong\holiday_judge_dict_2010_2018.pkl',
                       'rb+' )  # todo 注意读入文件的路径更改
        holiday_judge_dict = pickle.load ( file1 )

        tour_city = pd.read_csv ( r'C:\Users\Wangzhuo\Desktop\air2ndrong\tour_city_dict.csv', header=0 )
        tour_city_dict = tour_city.set_index ( 'IATA' ).T.to_dict ( 'records' )[0]
        city_class = pd.read_csv ( r'C:\Users\Wangzhuo\Desktop\air2ndrong\hlv_city_class_dict.csv', header=0 )
        city_class_dict = city_class.set_index ( 'IATA' ).T.to_dict ( 'records' )[0]

        def holiday_judge(x):
            try:
                return holiday_judge_dict[str ( x )[:10].replace ( '-', '' )]
            except:
                return 0

        def tour_city_judge(x):
            try:
                return tour_city_dict[x]
            except:
                return "not_tour"

        def city_class_judge(x):
            try:
                return city_class_dict[x]
            except:
                return 5

        air_df['arrv_day_type'] = air_df['seg_arrv_dt_lcl'].apply ( lambda x: holiday_judge ( x ) )
        air_df['tour_city_type_arr'] = air_df['arrival'].apply ( lambda x: tour_city_judge ( x ) )
        air_df['city_class_arr'] = air_df['arrival'].apply ( lambda x: city_class_judge ( x ) )
        air_df['tour_city_type_dep'] = air_df['departure'].apply ( lambda x: tour_city_judge ( x ) )
        air_df['city_class_dep'] = air_df['departure'].apply ( lambda x: city_class_judge ( x ) )
        air_df['depdate_tkt_bkg'] = air_df['depdate'] - air_df['tkt_bkg_dt']  # 计算提前订票间隔日期
        air_df['depdate_tkt_bkg'] = air_df['depdate_tkt_bkg'].apply ( lambda x: x.days )
        air_df['arrv_dep_level'] = air_df['city_class_arr'] - air_df['city_class_dep']
        air_df['seg_arrv_tl'] = air_df['seg_arrv_tm_lcl'].apply ( f_tbins )
        air_df['seg_dpt_tl'] = air_df['seg_dpt_tm_lcl'].apply ( f_tbins )
        air_df['cls_tl'] = air_df['cls_cd'].apply ( f_clsbins )
        air_df['arrv_dep_lvl'] = air_df['arrv_dep_level'].apply ( f_level_bins )
        return air_df

    air_df3 = prefeature_proces ( air_df2 )
    t_cnt = air_df3[data_date] - DateOffset ( months=t0 )
    air_df4 = air_df3[(air_df3[depdate] >= t_cnt) & (air_df3[data_date] > air_df3[depdate])]
    return air_df4


##一维特征衍生
def f_get1var(df_raw,col_index,t0,data_date,depdate,col,nm,func):
    """
    一维特征衍生函数，用于衍生单个特征，返回col_index+特征数据框
    :param df_raw: 输入数据框 必备列包括col_index和原始特征
    :param col_index:主键list,col_index=['UUID','data_date','target']
    :param t0: 回溯月份数，t0=12
    :param data_date: 回溯时间列，观察点
    :param depdate: 时间记录，用以计算回溯时长
    :param col: 衍生变量列
    :param nm: 衍生变量名，如'air_'
    :param func:计算方程
    :return:数据框，col_index+特征
    """
    t_cnt = df_raw[data_date] - DateOffset(months=t0)
    df_raw = df_raw[(df_raw[depdate] >= t_cnt) & (df_raw[data_date] > df_raw[depdate])]
    # vars_df=pd.DataFrame(df_raw[col_index].drop_duplicates()).reset_index()
    df_grouped=df_raw.groupby(col_index)
    # vars_df.set_index(col_index,inplace=True)
    vars_df=df_grouped[col].agg(func).reset_index()
    vars_df.columns=col_index+[nm+col+'_'+cl+str(t0)+'m' for cl in vars_df.columns[len(col_index):]]
    return vars_df
def f_vars1_df(df_raw,id,col_index,t0,data_date,depdate,nm,varsdict):
    """
    批量衍生特征
    :param df_raw:输入数据框 必备列包括col_index和原始特征，同f_get1var输入
    :param id:除时间、target外主键，如cid或UUID等
    :param col_index:主键list,col_index=['UUID','data_date','target']，同f_get1var输入
    :param t0:t0: 回溯月份数，t0=12，同f_get1var输入
    :param data_date:回溯时间列，观察点，同f_get1var输入
    :param depdate:时间记录，用以计算回溯时长，同f_get1var输入
    :param nm:衍生变量名，如'air_'，同f_get1var输入
    :param varsdict:字典，键为原始特征字段，值为衍生方程，为list
    varsdict={
    'depdate':['count'],
    'discount':['count',np.nanmax,np.nanmin,np.nanmedian,np.mean],
    'tkt_channel':['nunique'],
    'depdate_tkt_bkg':[np.nanmax,np.nanmin,np.nanmedian,np.mean]
}
    :return:col_index+批量特征，数据框
    """
    vars1_df=df_raw[col_index].drop_duplicates().sort_values(col_index,ascending=False)
    vars1_df.drop_duplicates([id,data_date],inplace=True)
    for var,func in varsdict.items():
        vars_add=f_get1var(df_raw,col_index,t0,data_date,depdate,var,nm,func)
        vars1_df=pd.merge(vars1_df,vars_add,on=col_index,how='left')
    return vars1_df


##二维特征衍生,one_hot特征
def f_long2wide(data,col_index,columns,value,f_):
    """
    数据框长转宽
    :param data:长数据框，详单数据，原始特征包含多个类别，该类别转为独立列
    :param col_index: 主键list,col_index=['UUID','data_date','target']
    :param columns: 包含多个类别的特征列
    :param value:一般为详单数据中的行为时间点，金额等
    :param f_:衍生方程,如np.size、np.sum等
    :return:宽数据框
    """
    df_cross=pd.pivot_table(data,index=col_index,columns=columns,values=value,aggfunc=f_).reset_index()
    return df_cross
def f_get2var(data,col_index,columns,value,func,t0,data_date,depdate,nm):
    """
    长转宽特征
    :param data: 长数据框，详单数据，原始特征包含多个类别，该类别转为独立列
    :param col_index:主键list,col_index=['UUID','data_date','target']
    :param columns:包含多个类别的特征列
    :param value:一般为详单数据中的行为时间点，金额等
    :param func:衍生方程,如np.size、np.sum等
    :param t0:回溯月份数，t0=12
    :param data_date:回溯时间列，观察点
    :param depdate:时间记录
    :param nm:衍生变量名，如'air_'
    :return:
    """
    t_cnt = data[data_date] - DateOffset(months=t0)
    df_raw = data[(data[depdate] >= t_cnt) & (data[data_date] > data[depdate])]
    vars2_df=f_long2wide(df_raw,col_index,columns,value,func)
    vars2_df.columns=col_index+[nm+columns+str(col)+'_cnts_'+str(t0)+'m' for col in vars2_df.columns[len(col_index):]]
    return vars2_df
def f_var2_df(df_raw,col_index,id,onehot_vars,value,t0,data_date,depdate,nm,func=np.size):
    """

    :param df_raw:长数据框，详单数据，原始特征包含多个类别，该类别转为独立列
    :param col_index:主键list,col_index=['UUID','data_date','target']
    :param id:除时间、target外主键，如cid或UUID等
    :param onehot_vars:
    one_hot_vars=['tour_city_type_arr','tour_city_type_dep','city_class_arr']

    :param value:一般为详单数据中的行为时间点，金额等
    :param t0:回溯月份数，t0=12
    :param data_date:回溯时间列，观察点
    :param depdate:时间记录，用以计算回溯时长
    :param nm:衍生变量名，如'air_'
    :param func:衍生方程,如np.size、np.sum等
    :return:
    """
    vars2_df=df_raw[col_index].drop_duplicates().sort_values(col_index,ascending=False)
    vars2_df.drop_duplicates([id,data_date],inplace=True)
    for var in onehot_vars:
        vars_onehot = f_get2var(df_raw, col_index, var, value, func, t0, data_date,depdate, nm)
        vars2_df=pd.merge(vars2_df,vars_onehot,on=col_index,how='left')
    return vars2_df


##最值特征衍生，如最近时间类特征，最值对应信息
def f_get_maxvars(df_raw,col_index,id,data_date,col,col_lst):
    """
    最值特征衍生函数
    :param df_raw: 输入数据框 必备列包括col_index和原始特征，同f_get1var输入
    :param col_index: 主键list,col_index=['UUID','data_date','target']
    :param id: 除时间、target外主键，如cid或UUID等
    :param data_date: 回溯时间列，观察点
    :param col: 原始特征，如金额、里程、日期等
    :param col_lst: 原始特征，list，用于衍生最值对应信息
    :return:
    """
    df_raw=df_raw.sort_values([id,col],ascending=False)
    df_lst=df_raw.drop_duplicates(col_index)
    if col=='depdate':
        ##衍生最近时长、最近一次行为对应信息等，如最近一次购买金额值
        lst_inf=df_lst[col_index+[col]+col_lst]
        lst_tt='lst_'+col+'_tt'
        lst_inf.columns=col_index+[col]+['lst_'+col+'_'+v for v in col_lst]
        lst_inf[lst_tt] = (lst_inf[data_date] - lst_inf[col]).apply(lambda x:x.days)
        lst_inf.drop([col],axis=1,inplace=True)
        return lst_inf
    else:
        ##衍生最大值对应市场信息
        lst_inf=df_lst[col_index+[col_lst]]
        var='max_'+col+'_tt'
        lst_inf[var]=(lst_inf[data_date]-lst_inf[col_lst]).apply(lambda x:x.days)
        return lst_inf
def f_getlst_var(df_raw,col_index,id,data_date,dt,max_lst):
    """
    批量最值特征衍生函数
    :param df_raw: 输入数据框 必备列包括col_index和原始特征，同f_get1var输入
    :param col_index: 主键list,col_index=['UUID','data_date','target']
    :param id: 除时间、target外主键，如cid或UUID等
    :param data_date: 回溯时间列，观察点
    :param dt: 时间记录,同前函数中的depdate
    :param max_lst:最大值对应原始特征list
    max_lst=['yprice','segprice','tkt_amt']
    :return:
    """
    maxvar_tt_df=df_raw[col_index].drop_duplicates().sort_values(col_index,ascending=False)
    maxvar_tt_df.drop_duplicates([id,data_date],inplace=True)
    for mv in max_lst:
        maxvar_tt_tmp = f_get_maxvars(df_raw, col_index, id, data_date, mv, dt)
        maxvar_tt_df=pd.merge(maxvar_tt_df,maxvar_tt_tmp,on=col_index,how='left')
    var_lst_tt = f_get_maxvars(df_raw, col_index, id, data_date, dt, max_lst)
    vars_lst=pd.merge(maxvar_tt_df,var_lst_tt,on=col_index,how='left')
    col_lst=col_index+[col for col in vars_lst.columns if col.find('lst')!=-1 or col.find('max')!=-1]
    vars_lst=vars_lst[col_lst]
    return vars_lst


##斜率特征，观察期内变化率
def f_yx(df_raw,data_date,dt,y,col_index):
    """
    斜率特征衍生，变化率
    :param df_raw:输入数据框 必备列包括col_index和原始特征，同f_get1var输入
    :param data_date: 回溯时间列,观察点
    :param dt: 详单时间点，同之前函数中的depdate
    :param y: 用于计算斜率的原始特征，如金额等
    :param col_index: 主键list,col_index=['UUID','data_date','target']
    :return:
    """
    df=df_raw[pd.notna(df_raw[y])][col_index+[y,dt]]
    df['rk'] = (df[data_date] - df[dt]).apply(lambda x: x.days)
    df_ymean=df.groupby(col_index)[y].mean().reset_index().rename(columns={y:y+'_mean'})
    df_ym=pd.merge(df,df_ymean,how='left')
    df_ym['y_m']=df_ym[y]-df_ym[y+'_mean']
    df_xmean=df.groupby(col_index)['rk'].mean().reset_index().rename(columns={'rk':'rk_mean'})
    df_xym=pd.merge(df_ym,df_xmean,how='left')
    df_xym['rk_m']=df_xym['rk']-df_xym['rk_mean']
    df_xym['rk_m2']=df_xym['rk_m'].apply(lambda x:x*x)
    df_xym['xy']=df_xym['y_m']*df_xym['rk_m']
    df_rt=(df_xym.groupby(col_index)['xy'].sum()/df_xym.groupby(col_index)['rk_m2'].sum()).reset_index()
    df_rt.columns=col_index+[y+'_rt']
    # df_rt=df_rt.sort_values(col_index,ascending=False)
    # df_rt.drop_duplicates(id,inplace=True)
    return df_rt

##连续天数、时间间隔特征衍生
def f_sequedays_vars(df_raw,id,dt,cols_index):
    """
    用于连续天数、时间间隔特征衍生，包括最大连续天数，最大、最小等时间间隔
    :param df_raw: 输入数据框 必备列包括col_index和原始特征，同f_get1var输入
    :param id: 除时间、target外主键，如cid或UUID等
    :param dt: 详单时间点，同之前函数中的depdate
    :param cols_index: 主键list,col_index=['UUID','data_date','target']
    :return:
    """
    df_raw['date_rank'] = df_raw.groupby ( [id] )[dt].rank ().astype ( int )
    f_days_trans = lambda x: timedelta ( x )
    df_raw['date_rank'] = df_raw['date_rank'].apply ( f_days_trans )
    df_raw['date_diff'] = df_raw[dt] - df_raw['date_rank']
    df_raw['seque_cnt'] = 1
    daysdiff_sequece = df_raw.groupby ( cols_index + ['date_diff'] )[
        'seque_cnt'].count ().reset_index ()
    daysdiff_maxsequece=daysdiff_sequece.groupby( cols_index )['seque_cnt'].agg( np.nanmax ).reset_index()
    vars_maxsqcnt='seque_cnt'+'_max'
    daysdiff_maxsequece.columns=cols_index+[vars_maxsqcnt]
    df_raw = df_raw.sort_values ( [dt] )
    df_raw['dt_lead'] = df_raw.groupby ( id )[dt].shift ()
    df_raw['days_diff'] = (df_raw[dt] - df_raw['dt_lead']).apply (
        lambda x: x.days if pd.notnull ( x ) else np.nan )
    daysdiff_min = df_raw.groupby ( cols_index )['days_diff'].agg ( np.nanmin ).reset_index ()
    var_min = 'days_diff' + '_min'
    daysdiff_min.columns = cols_index + [var_min]
    daysdiff_max = df_raw.groupby ( cols_index )['days_diff'].agg ( np.nanmax ).reset_index ()
    var_max = 'days_diff' + '_max'
    daysdiff_max.columns = cols_index + [var_max]
    daysdiff_mean = df_raw.groupby ( cols_index )['days_diff'].agg ( np.nanmean ).reset_index ()
    var_mean = 'days_diff' + '_mean'
    daysdiff_mean.columns = cols_index + [var_mean]
    return daysdiff_maxsequece,daysdiff_min,daysdiff_max,daysdiff_mean


##特征合并
def f_merge(x,y):
    return pd.merge(x,y,on=col_index,how='outer')
def f_merge_all(vars_lst):
    return reduce(f_merge,vars_lst)

if __name__=="main":
    col_index = ['UUID', 'data_date', 'target']
    t1 = np.datetime64 ( datetime ( 1900, 1, 1, 7, 0, 0 ) )
    t2 = np.datetime64 ( datetime ( 1900, 1, 1, 12, 0, 0 ) )
    t3 = np.datetime64 ( datetime ( 1900, 1, 1, 17, 0, 0 ) )
    t4 = np.datetime64 ( datetime ( 1900, 1, 1, 22, 0, 0 ) )

    air_df0 = pd.read_csv ( r'C:\Users\Wangzhuo\Desktop\air2ndrong\ccx_hl_df.csv', encoding='utf-8' )

    ##数据预处理,根据实际情况而定
    air_df_prc = data_process ( air_df0, 'data_date', 'depdate', 12 )
    air_df_raw = air_df_prc[air_df_prc.noshow_ind == 0]

    # 一维统计特征及函数列表，用于衍生特质
    varsdict_inf = {
        '飞行次数': 'depdate',  # 'count'
        '折扣次数': 'discount',  # 'count'
        '订票渠道数': 'tkt_channel',  # 'nunique'
        '常客卡数': 'ffp_card_cd',  # 'nunique'
        '支付渠道数': 'tkt_payment_typ',  # 'nunique'
        '最大折扣': 'discount',  # 'np.nanmax'
        '最小折扣': 'discount',  # 'np.nanmin'
        '平均折扣': 'discount',  # 'np.nanmean'
        '折扣中位数': 'discount',  # 'np.nanmedian'
        '最大订票金额': 'tkt_amt',  # 'np.nanmax'
        '最小订票金额': 'tkt_amt',  # 'np.nanmin'
        '平均订票金额': 'tkt_amt',  # 'np.nanmean'
        '订票金额中位数': 'tkt_amt',  # 'np.nanmedian'
        '累计订票金额': 'tkt_amt',  # 'np.nansum'
        '最大航段价格': 'segprice',  # 'np.nanmax'
        '最小航段价格': 'segprice',  # 'np.nanmin'
        '平均航段价格': 'segprice',  # 'np.nanmean'
        '航段价格中位数': 'segprice',  # 'np.nanmedian'
        '最大里程数': 'mileage',  # 'np.nanmax'
        '最小里程数': 'mileage',  # 'np.nanmin'
        '平均里程数': 'mileage',  # 'np.nanmean'
        '里程数中位数': 'mileage',  # 'np.nanmedian'
        '累计里程数': 'mileage',  # 'np.nansum'
        '最大托运重量': 'chkd_bag_wgt_qty',  # 'np.nanmax'
        '最小托运重量': 'chkd_bag_wgt_qty',  # 'np.nanmin'
        '平均托运重量': 'chkd_bag_wgt_qty',  # 'np.nanmean'
        '托运重量中位数': 'chkd_bag_wgt_qty',  # 'np.nanmedian'
        '累计托运件重量': 'chkd_bag_wgt_qty',  # 'np.nansum'
        '最大托运数量': 'bag_pcs',  # 'np.nanmax'
        '最小托运数量': 'bag_pcs',  # 'np.nanmin'
        '平均托运数量': 'bag_pcs',  # 'np.nanmean'
        '托运数量中位数': 'bag_pcs',  # 'np.nanmedian'
        '累计托运件数': 'bag_pcs',  # 'np.nansum'
        '最大经济舱价格': 'yprice',  # 'np.nanmax'
        '最小经济舱价格': 'yprice',  # 'np.nanmin'
        '平均经济舱价格': 'yprice',  # 'np.nanmean'
        '经济舱价格中位数': 'yprice',  # 'np.nanmedian'
        '最大订票天数': 'depdate_tkt_bkg',  # 'np.nanmax'
        '最小订票天数': 'depdate_tkt_bkg',  # 'np.nanmin'
        '平均订票天数': 'depdate_tkt_bkg',  # 'np.nanmean'
        '订票天数中位数': 'depdate_tkt_bkg',  # 'np.nanmedian'
        '最高出发到达机场等级差': 'arrv_dep_lvl'  # 'np.nanmax'
    }
    varsdict = {
        'depdate': ['count'],
        'discount': ['count', np.nanmax, np.nanmin, np.nanmedian, np.mean],
        'tkt_channel': ['nunique'],
        'ffp_card_cd': ['nunique'],
        'tkt_payment_typ': ['nunique'],
        'tkt_amt': [np.nanmax, np.nanmin, np.nanmedian, np.mean],
        'segprice': [np.nanmax, np.nanmin, np.nanmedian, np.mean],
        'mileage': [np.nanmax, np.nanmin, np.nanmedian, np.mean, np.nansum],
        'chkd_bag_wgt_qty': [np.nanmax, np.nanmin, np.nanmedian, np.mean, np.nansum],
        'bag_pcs': [np.nanmax, np.nanmin, np.nanmedian, np.mean, np.nansum],
        'yprice': [np.nanmax, np.nanmin, np.nanmedian, np.mean],
        # 'arrv_dep_lvl': [np.nanmax],
        'depdate_tkt_bkg': [np.nanmax, np.nanmin, np.nanmedian, np.mean]
    }


    vars12_df1 = f_vars1_df ( air_df_raw, 'UUID', col_index, 12, 'data_date', 'depdate', 'air_', varsdict )
    vars6_df1 = f_vars1_df ( air_df_raw, 'UUID', col_index, 6, 'data_date', 'depdate', 'air_', varsdict )
    vars12_df1_noshow = f_vars1_df ( air_df_prc[air_df_prc.noshow_ind == 1], 'UUID', col_index, 12, 'data_date',
                                     'depdate', 'air_noshow_', {'depdate': ['count']} )
    vars6_df1_noshow = f_vars1_df ( air_df_prc[air_df_prc.noshow_ind == 1], 'UUID', col_index, 6, 'data_date',
                                    'depdate', 'air_noshow_', {'depdate': ['count']} )
    vars12_df1_dom = f_vars1_df ( air_df_raw[air_df_raw['seg_typ'] == 'D'], 'UUID', col_index, 12, 'data_date',
                                  'depdate', 'air_dom_', {'depdate': ['count']} )
    vars6_df1_dom = f_vars1_df ( air_df_raw[air_df_raw['seg_typ'] == 'D'], 'UUID', col_index, 6, 'data_date', 'depdate',
                                 'air_dom_', {'depdate': ['count']} )
    vars12_df1_weekday = f_vars1_df ( air_df_raw[air_df_raw.arrv_day_type == 0], 'UUID', col_index, 12, 'data_date',
                                      'depdate', 'air_', varsdict )
    vars12_df1_weekday.columns = col_index + [col + '_wkday' for col in vars12_df1_weekday.columns[len ( col_index ):]]
    vars6_df1_weekday = f_vars1_df ( air_df_raw[air_df_raw.arrv_day_type == 0], 'UUID', col_index, 6, 'data_date',
                                     'depdate', 'air_', varsdict )
    vars6_df1_weekday.columns = col_index + [col + '_wkday' for col in vars6_df1_weekday.columns[len ( col_index ):]]
    vars12_df1_weekend = f_vars1_df ( air_df_raw[air_df_raw.arrv_day_type == 1], 'UUID', col_index, 12, 'data_date',
                                      'depdate', 'air_', varsdict )
    vars12_df1_weekend.columns = col_index + [col + '_wkend' for col in vars12_df1_weekend.columns[len ( col_index ):]]
    vars6_df1_weekend = f_vars1_df ( air_df_raw[air_df_raw.arrv_day_type == 1], 'UUID', col_index, 6, 'data_date',
                                     'depdate', 'air_', varsdict )
    vars6_df1_weekend.columns = col_index + [col + '_wkend' for col in vars6_df1_weekend.columns[len ( col_index ):]]
    vars12_df1_holiday = f_vars1_df ( air_df_raw[air_df_raw.arrv_day_type == 2], 'UUID', col_index, 12, 'data_date',
                                      'depdate', 'air_', varsdict )
    vars12_df1_holiday.columns = col_index + [col + '_hlday' for col in vars12_df1_holiday.columns[len ( col_index ):]]
    vars6_df1_holiday = f_vars1_df ( air_df_raw[air_df_raw.arrv_day_type == 2], 'UUID', col_index, 6, 'data_date',
                                     'depdate', 'air_', varsdict )
    vars6_df1_holiday.columns = col_index + [col + '_hlday' for col in vars6_df1_holiday.columns[len ( col_index ):]]

    ##二维特征衍生,one_hot特征
    one_hot_vars = ['tour_city_type_arr', 'tour_city_type_dep', 'city_class_arr', 'city_class_dep', 'cls_tl',
                    'seg_arrv_tl', 'seg_dpt_tl', 'arrv_dep_lvl']


    vars12_df2 = f_var2_df ( air_df_raw, col_index, 'UUID', one_hot_vars, 'depdate', 12, 'data_date', 'depdate',
                             'air_' )
    vars6_df2 = f_var2_df ( air_df_raw, col_index, 'UUID', one_hot_vars, 'depdate', 6, 'data_date', 'depdate', 'air_' )
    vars12_df2_weekday = f_var2_df ( air_df_raw[air_df_raw.arrv_day_type == 0], col_index, 'UUID', one_hot_vars,
                                     'depdate', 12, 'data_date', 'depdate', 'air_' )
    vars12_df2_weekday.columns = col_index + [col + 'wkday' for col in vars12_df2_weekday.columns[len ( col_index ):]]
    vars6_df2_weekday = f_var2_df ( air_df_raw[air_df_raw.arrv_day_type == 0], col_index, 'UUID', one_hot_vars,
                                    'depdate', 6, 'data_date', 'depdate', 'air_' )
    vars6_df2_weekday.columns = col_index + [col + 'wkday' for col in vars6_df2_weekday.columns[len ( col_index ):]]
    vars12_df2_weekend = f_var2_df ( air_df_raw[air_df_raw.arrv_day_type == 1], col_index, 'UUID', one_hot_vars,
                                     'depdate', 12, 'data_date', 'depdate', 'air_' )
    vars12_df2_weekend.columns = col_index + [col + 'wkend' for col in vars12_df2_weekend.columns[len ( col_index ):]]
    vars6_df2_weekend = f_var2_df ( air_df_raw[air_df_raw.arrv_day_type == 1], col_index, 'UUID', one_hot_vars,
                                    'depdate', 6, 'data_date', 'depdate', 'air_' )
    vars6_df2_weekend.columns = col_index + [col + 'wkend' for col in vars6_df2_weekend.columns[len ( col_index ):]]
    vars12_df2_holiday = f_var2_df ( air_df_raw[air_df_raw.arrv_day_type == 2], col_index, 'UUID', one_hot_vars,
                                     'depdate', 12, 'data_date', 'depdate', 'air_' )
    vars12_df2_holiday.columns = col_index + [col + 'hlday' for col in vars12_df2_holiday.columns[len ( col_index ):]]
    vars6_df2_holiday = f_var2_df ( air_df_raw[air_df_raw.arrv_day_type == 2], col_index, 'UUID', one_hot_vars,
                                    'depdate', 6, 'data_date', 'depdate', 'air_' )
    vars6_df2_holiday.columns = col_index + [col + 'hlday' for col in vars6_df2_holiday.columns[len ( col_index ):]]


    ##最值统计信息，包括最近行为信息、最值对应信息等，主要基于二维groupby
    max_lst = ['yprice', 'segprice', 'tkt_amt']
    vars4lst = f_getlst_var ( air_df_raw, col_index, 'UUID', 'data_date', 'depdate', max_lst )
    vars4lst_weekday = f_getlst_var ( air_df_raw[air_df_raw.arrv_day_type == 0], col_index, 'UUID', 'data_date',
                                      'depdate', max_lst )
    vars4lst_weekday.columns = col_index + [col + '_wkday' for col in vars4lst_weekday.columns[len ( col_index ):]]
    vars4lst_weekend = f_getlst_var ( air_df_raw[air_df_raw.arrv_day_type == 1], col_index, 'UUID', 'data_date',
                                      'depdate', max_lst )
    vars4lst_weekend.columns = col_index + [col + '_wkend' for col in vars4lst_weekend.columns[len ( col_index ):]]
    vars4lst_holiday = f_getlst_var ( air_df_raw[air_df_raw.arrv_day_type == 2], col_index, 'UUID', 'data_date',
                                      'depdate', max_lst )
    vars4lst_holiday.columns = col_index + [col + '_hlday' for col in vars4lst_holiday.columns[len ( col_index ):]]

    ##斜率特征，观察期内变化率
    var_yx_tkt = f_yx ( air_df_raw[pd.notna ( air_df_raw.tkt_amt )], 'data_date', 'depdate', 'tkt_amt', col_index )
    var_yx_seg = f_yx ( air_df_raw[pd.notna ( air_df_raw.segprice )], 'data_date', 'depdate', 'segprice', col_index )
    var_yx_yprice = f_yx ( air_df_raw[pd.notna ( air_df_raw.yprice )], 'data_date', 'depdate', 'yprice', col_index )

    ##连续天数、时间间隔特征衍生
    vars_daysdiff_maxsq, vars_daysdiff_min, vars_daysdiff_max, vars_daysdiff_mean = f_sequedays_vars ( air_df_raw,
                                                                                                       'UUID',
                                                                                                       'depdate',
                                                                                                       col_index )

    ##特征合并
    vars_lst = [vars12_df1, vars6_df1, vars12_df1_dom, vars6_df1_dom, vars12_df1_noshow, vars6_df1_noshow,
                vars12_df1_weekday, vars6_df1_weekday,
                vars12_df1_weekend, vars6_df1_weekend, vars12_df1_holiday, vars6_df1_holiday,
                vars12_df2, vars6_df2, vars12_df2_weekday, vars6_df2_weekday,
                vars12_df2_weekend, vars6_df2_weekend, vars12_df2_holiday, vars6_df2_holiday,
                vars4lst, vars4lst_weekday, vars4lst_weekend, vars4lst_holiday, var_yx_tkt, var_yx_seg,
                vars_daysdiff_maxsq, vars_daysdiff_min, vars_daysdiff_max, vars_daysdiff_mean]

    vars_all = f_merge_all(vars_lst)



vars_all.to_csv('features_df.csv')