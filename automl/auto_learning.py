from hyperopt import fmin,tpe,hp,partial,STATUS_OK
from hyperopt import Trials as hp_trials
from model_results import *
from sklearn.model_selection import GridSearchCV
import optuna
from utils import *


# def test_function(paras):
#     global data_sets
#     global model
#     estmator=model(**paras)
#     estmator.fit(data_sets['train_x'],data_sets['train_y'])
#     test_r2=regress_results(data_sets['test_x'],data_sets['test_y'],estmator)[0].iloc[0,1]
#     future_r2=regress_results(data_sets['data_future'],data_sets['target_future'],estmator)[0].iloc[0,1]
#     print(paras)
#     print('future_r2:',future_r2)
#     return 1-test_r2

# def future_function(paras):
#     global data_sets
#     estmator=model(**paras,random_state=28)
#     estmator.fit(data_sets['train_x'],data_sets['train_y'])
#     future_r2=regress_results(data_sets['data_future'],data_sets['target_future'],estmator)[0].iloc[0,1]
#     print(paras)
#     return 1-future_r2

logger=get_logger()



class hyper_best:
    """
    data_sets为字典,'train_x':train_x，
    当function_data输入为train_x,train_y等，data_sets={}
    """
    def __init__(self,data_sets,paras_input):
        self.trials = hp_trials()
        self.data_sets=data_sets
        self.model=paras_input['model']
        self.random_state=paras_input['random_state']

        if paras_input['time_splt']==1:
            self.ks_flag='test'
            self.auc_flag='test'
            self.regress_flag='test'
        else:
            self.ks_flag='future'
            self.auc_flag='future'
            self.regress_flag='future'

    def hp_space(self,space):
        space4model= self.f_spacehp(space)
        return space4model

    def regressor_fmin(self,space4model):
        test_best=fmin(self.regressor,space4model,algo=partial(tpe.suggest,n_startup_jobs=1),max_evals=100,trials=self.trials)
        return test_best

    def auc_fmin(self,space4model):
        test_best=fmin(self.auc,space4model,algo=partial(tpe.suggest,n_startup_jobs=1),max_evals=100,trials=self.trials)
        return test_best

    def ks_fmin(self,space4model):
        test_best=fmin(self.ks,space4model,algo=partial(tpe.suggest,n_startup_jobs=1),max_evals=100,trials=self.trials)
        return test_best

    def bestparam(self,test_best,space):
        best_param = self.f_bestparam(test_best,space)
        return best_param

    def regressor_best(self,space):
        space_4model = self.f_spacehp(space)
        test4best = fmin(self.regressor,space_4model,algo=partial(tpe.suggest,n_startup_jobs=1),max_evals=100,trials=self.trials)
        best_params = self.f_bestparam(test4best,space)
        best_model = self.model(**best_params, random_state=self.random_state)
        best_model.fit(self.data_sets['train_x'], self.data_sets['train_y'])
        try:
            train_r2 = regress_results(self.data_sets['train_x'], self.data_sets['train_y'], best_model)[0].iloc[0, 1]
        except:
            train_r2 = None
        try:
            test_r2 = regress_results(self.data_sets['test_x'], self.data_sets['test_y'], best_model)[0].iloc[0, 1]
        except:
            test_r2 = None
        try:
            future_r2 = regress_results(self.data_sets['data_future'], self.data_sets['target_future'], best_model)[0].iloc[0, 1]
        except:
            future_r2 = None

        best_results = {'train_r2': train_r2, 'test_r2': test_r2, 'future_r2': future_r2}

        return best_params,best_model,best_results

    def auc_best(self,space):
        space_4model = self.f_spacehp(space)
        test4best = fmin(self.auc,space_4model,algo=partial(tpe.suggest,n_startup_jobs=1),max_evals=100,trials=self.trials)
        best_params = self.f_bestparam(test4best,space)

        best_model = self.model(**best_params, random_state=self.random_state)
        best_model.fit(self.data_sets['train_x'], self.data_sets['train_y'])

        try:
            train_ks = ks_results(self.data_sets['train_x'], self.data_sets['train_y'], best_model)
        except:
            train_ks = None
        try:
            test_ks = ks_results(self.data_sets['test_x'], self.data_sets['test_y'], best_model)
        except:
           test_ks = None
        try:
            future_ks = ks_results(self.data_sets['data_future'], self.data_sets['target_future'], best_model)
        except:
            future_ks = None
        try:
            train_auc = auc_results(self.data_sets['train_x'], self.data_sets['train_y'], best_model)
        except:
            train_auc = None
        try:
            test_auc = auc_results(self.data_sets['test_x'], self.data_sets['test_y'], best_model)
        except:
            test_auc = None
        try:
            future_auc = auc_results(self.data_sets['data_future'], self.data_sets['target_future'], best_model)
        except:
            future_auc = None

        best_results = {'train_auc': train_auc,'test_auc': test_auc, 'future_auc': future_auc,'train_ks': train_ks, 'test_ks': test_ks, 'future_ks': future_ks}

        return best_params, best_model, best_results


    def ks_best(self,space):
        space_4model = self.f_spacehp(space)
        test4best = fmin(self.ks,space_4model,algo=partial(tpe.suggest,n_startup_jobs=1),max_evals=100,trials=self.trials)
        best_params = self.f_bestparam(test4best,space)

        best_model = self.model(**best_params,random_state=self.random_state)
        best_model.fit(self.data_sets['train_x'], self.data_sets['train_y'])

        try:
            train_ks = ks_results(self.data_sets['train_x'], self.data_sets['train_y'], best_model)
        except:
            train_ks = None
        try:
            test_ks = ks_results(self.data_sets['test_x'], self.data_sets['test_y'], best_model)
        except:
            test_ks = None
        try:
            future_ks = ks_results(self.data_sets['data_future'], self.data_sets['target_future'], best_model)
        except:
            future_ks = None
        try:
            train_auc = auc_results(self.data_sets['train_x'], self.data_sets['train_y'], best_model)
        except:
            train_auc = None
        try:
            test_auc = auc_results(self.data_sets['test_x'], self.data_sets['test_y'], best_model)
        except:
            test_auc = None
        try:
            future_auc = auc_results(self.data_sets['data_future'], self.data_sets['target_future'], best_model)
        except:
            future_auc = None

        best_results = {'train_auc': train_auc,'test_auc': test_auc, 'future_auc': future_auc,'train_ks': train_ks, 'test_ks': test_ks, 'future_ks': future_ks}


        return best_params, best_model, best_results


    def ks(self,paras):
        logger.info('One Step of Model Training...')
        estmator = self.model(**paras,random_state=self.random_state)
        estmator.fit(self.data_sets['train_x'], self.data_sets['train_y'])
        try:
            train_ks = ks_results(self.data_sets['train_x'], self.data_sets['train_y'], estmator)[0].iloc[0, 1]
        except:
            train_ks = None
        try:
            test_ks = ks_results(self.data_sets['test_x'], self.data_sets['test_y'], estmator)[0].iloc[0, 1]
        except:
            test_ks = None
        try:
            future_ks = ks_results(self.data_sets['data_future'], self.data_sets['target_future'], estmator)[0].iloc[0, 1]
        except:
            future_ks = None
        logger.info('current_params: \n{}'.format(paras))
        logger.info('train_ks: {}'.format(train_ks))
        logger.info('test_ks: {}'.format(test_ks))
        logger.info('future_ks: {}'.format(future_ks))

        if self.ks_flag=='test':
            return 1 - test_ks
        else:
            return 1-future_ks

    def ks_data(self,paras,train_x,train_y,test_x,test_y,data_future,target_future):
        logger.info('One Step of Model Training...')
        estmator = self.model(**paras,random_state=self.rf)
        estmator.fit(train_x, train_y)
        train_ks = ks_results(test_x, test_y, estmator)[0].iloc[0, 1]
        test_ks = ks_results(test_x, test_y, estmator)[0].iloc[0, 1]
        future_ks = ks_results(data_future, target_future, estmator)[0].iloc[0, 1]
        logger.info('current_params: \n{}'.format(paras))
        logger.info('train_ks: {}'.format(train_ks))
        logger.info('test_ks: {}'.format(test_ks))
        logger.info('future_ks: {}'.format(future_ks))
        return 1 - test_ks

    def auc(self,paras):
        logger.info('One Step of Model Training...')
        estmator = self.model(**paras,random_state=self.random_state)
        estmator.fit(self.data_sets['train_x'], self.data_sets['train_y'])
        try:
            train_auc = auc_results(self.data_sets['train_x'], self.data_sets['train_y'], estmator)
        except:
            train_auc = None
        try:
            test_auc = auc_results(self.data_sets['test_x'], self.data_sets['test_y'], estmator)
        except:
            test_auc = None
        try:
            future_auc = auc_results(self.data_sets['data_future'], self.data_sets['target_future'], estmator)#[0].iloc[0, 1]
        except:
            future_auc = None

        logger.info('current_params: \n{}'.format(paras))
        logger.info('train_auc: {}'.format(train_auc))
        logger.info('test_auc: {}'.format(test_auc))
        logger.info('future_auc: {}'.format(future_auc))
        if self.auc_flag=='test':
            return 1 - test_auc
        else:
            return 1-future_auc

    def auc_data(self,paras,train_x,train_y,test_x,test_y,data_future,target_future):
        estmator = self.model(**paras,random_state=self.random_state)
        estmator.fit(train_x, train_y)
        train_auc = auc_results(test_x, test_y, estmator)#[0].iloc[0, 1]
        test_auc = auc_results(test_x, test_y, estmator)#[0].iloc[0, 1]
        future_auc = auc_results(data_future, target_future, estmator)#[0].iloc[0, 1]
        logger.info('current_params: \n{}'.format(paras))
        logger.info('train_auc: {}'.format(train_auc))
        logger.info('test_auc: {}'.format(test_auc))
        logger.info('future_auc: {}'.format(future_auc))
        return 1 - test_auc

    def regressor(self,paras):
        logger.info('One Step of Model Training...')
        estmator = self.model(**paras,random_state=self.random_state)
        estmator.fit(self.data_sets['train_x'], self.data_sets['train_y'])
        try:
            train_r2 = regress_results(self.data_sets['train_x'], self.data_sets['train_y'], estmator)[0].iloc[0, 1]
        except:
            train_r2 = None
        try:
            test_r2 = regress_results(self.data_sets['test_x'], self.data_sets['test_y'], estmator)[0].iloc[0, 1]
        except:
            test_r2 = None
        try:
            future_r2 = regress_results(self.data_sets['data_future'], self.data_sets['target_future'], estmator)[0].iloc[0, 1]
        except:
            future_r2 = None
        logger.info('current_params: \n{}'.format(paras))
        logger.info('train_r2: {}'.format(train_r2))
        logger.info('test_r2: {}'.format(test_r2))
        logger.info('future_r2: {}'.format(future_r2))

        if self.regress_flag=='test':
            return 1 - test_r2
        else:
            return 1 - future_r2

    def regressor_data(self,paras,train_x,train_y,test_x,test_y,data_future,target_future):
        logger.info('One Step of Model Training...')
        estmator = self.model(**paras,random_state=self.random_state)
        estmator.fit(train_x, train_y)
        train_r2 = regress_results(train_x, train_y, estmator)[0].iloc[0, 1]
        test_r2 = regress_results(test_x, test_y, estmator)[0].iloc[0, 1]
        future_r2 = regress_results(data_future, target_future, estmator)[0].iloc[0, 1]
        logger.info('current_params: \n{}'.format(paras))
        logger.info('train_r2: {}'.format(train_r2))
        logger.info('test_r2: {}'.format(test_r2))
        logger.info('future_r2: {}'.format(future_r2))
        return 1 - test_r2



    def f_spacehp(self,space):
        space4rf = dict()
        for k,v in space.items():
            space4rf[k]=hp.choice(k,space[k])
        return space4rf

    def f_bestparam(self,test_best,space):
        best_param=dict()
        for ki,vi in test_best.items():
            param_tmp=space[ki][test_best[ki]]
            best_param[ki]=param_tmp
        return best_param



# def objective(trial,para_dct,model):
#     op_dct = {}
#     for k, v in para_dct.items():
#            op_dct[k] = trial.suggest_categorical(k, v)
#
#     clf=model(**op_dct)
#     clf.fit(data_sets['train_x'], data_sets['train_y'])
#
#     test_auc=auc_results(data_sets['test_x'], data_sets['test_y'], clf)
#
#     return 1-test_auc


class optu_best:
    def __init__(self,data_sets,paras_input):
        self.data_sets=data_sets
        self.model=paras_input['model']
        self.para_dct=paras_input['model_paras']
        self.random_state=paras_input['random_state']

        if paras_input['time_splt']==1:
            self.ks_flag='test'
            self.auc_flag='test'
            self.regress_flag='test'
        else:
            self.ks_flag='future'
            self.auc_flag='future'
            self.regress_flag='future'

    def ks_objective(self,trial):
        op_dct = {}
        for k, v in self.para_dct.items():
            op_dct[k] = trial.suggest_categorical(k, v)

        clf = self.model(**op_dct,random_state=self.random_state)
        clf.fit(self.data_sets['train_x'], self.data_sets['train_y'])

        try:
            train_ks = ks_results(self.data_sets['train_x'], self.data_sets['train_y'], clf)
        except:
            train_ks = None
        try:
            test_ks = ks_results(self.data_sets['test_x'], self.data_sets['test_y'], clf)
        except:
            test_ks = None
        try:
            future_ks = ks_results(self.data_sets['data_future'], self.data_sets['target_future'], clf)
        except:
            future_ks = None

        logger.info('current_params: \n{}'.format(op_dct))
        logger.info('train_ks: {}'.format(train_ks))
        logger.info('test_ks: {}'.format(test_ks))
        logger.info('future_ks: {}'.format(future_ks))

        if self.ks_flag == 'test':
            return 1 - test_ks
        else:
            return 1 - future_ks

    def auc_objective(self, trial):
        op_dct = {}
        for k, v in self.para_dct.items():
            op_dct[k] = trial.suggest_categorical(k, v)

        clf = self.model(**op_dct, random_state=self.random_state)
        clf.fit(self.data_sets['train_x'], self.data_sets['train_y'])

        try:
            train_auc = auc_results(self.data_sets['train_x'], self.data_sets['train_y'], clf)
        except:
            train_auc = None
        try:
            test_auc = auc_results(self.data_sets['test_x'], self.data_sets['test_y'], clf)
        except:
            test_auc = None
        try:
            future_auc = auc_results(self.data_sets['data_future'], self.data_sets['target_future'], clf)
        except:
            future_auc = None
        logger.info('\n')
        logger.info('current_params: \n{}'.format(op_dct))
        logger.info('train_auc: {}'.format(train_auc))
        logger.info('test_auc: {}'.format(test_auc))
        logger.info('future_auc: {}'.format(future_auc))
        if self.auc_flag == 'test':
            return 1 - test_auc
        else:
            return 1 - future_auc

    def ks_opt(self,n_trials):
        study = optuna.create_study()  # Create a new study.
        study.optimize(self.ks_objective, n_trials=n_trials)
        best_paras=study.best_params
        best_clf=self.model(**best_paras, random_state=self.random_state)
        best_clf.fit(self.data_sets['train_x'], self.data_sets['train_y'])
        try:
            train_ks = ks_results(self.data_sets['train_x'], self.data_sets['train_y'], best_clf)
        except:
            train_ks = None
        try:
            test_ks = ks_results(self.data_sets['test_x'], self.data_sets['test_y'], best_clf)
        except:
            test_ks = None
        try:
            future_ks = ks_results(self.data_sets['data_future'], self.data_sets['target_future'], best_clf)
        except:
            future_ks = None
        try:
            train_auc = auc_results(self.data_sets['train_x'], self.data_sets['train_y'], best_clf)
        except:
            train_auc = None
        try:
            test_auc = auc_results(self.data_sets['test_x'], self.data_sets['test_y'], best_clf)
        except:
            test_auc = None
        try:
            future_auc = auc_results(self.data_sets['data_future'], self.data_sets['target_future'], best_clf)
        except:
            future_auc = None

        best_results = {'train_auc': train_auc,'test_auc': test_auc, 'future_auc': future_auc,
                        'train_ks': train_ks, 'test_ks': test_ks, 'future_ks': future_ks}


        return best_paras,best_clf,best_results

    def auc_opt(self,n_trials=100):
        study = optuna.create_study()  # Create a new study.
        study.optimize(self.auc_objective, n_trials=n_trials)
        best_paras=study.best_params
        best_clf=self.model(**best_paras, random_state=self.random_state)
        best_clf.fit(self.data_sets['train_x'], self.data_sets['train_y'])
        try:
            train_ks = ks_results(self.data_sets['train_x'], self.data_sets['train_y'], best_clf)
        except:
            train_ks = None
        try:
            test_ks = ks_results(self.data_sets['test_x'], self.data_sets['test_y'], best_clf)
        except:
            test_ks = None
        try:
            future_ks = ks_results(self.data_sets['data_future'], self.data_sets['target_future'], best_clf)
        except:
            future_ks = None
        try:
            train_auc = auc_results(self.data_sets['train_x'], self.data_sets['train_y'], best_clf)
        except:
            train_auc = None
        try:
            test_auc = auc_results(self.data_sets['test_x'], self.data_sets['test_y'], best_clf)
        except:
            test_auc = None
        try:
            future_auc = auc_results(self.data_sets['data_future'], self.data_sets['target_future'], best_clf)
        except:
            future_auc = None

        best_results = {'train_auc': train_auc,'test_auc': test_auc, 'future_auc': future_auc,
                        'train_ks': train_ks, 'test_ks': test_ks, 'future_ks': future_ks}

        return best_paras, best_clf, best_results

    def regress_objective(self,trial):
        op_dct = {}
        for k, v in self.para_dct.items():
            op_dct[k] = trial.suggest_categorical(k, v)

        clf = self.model(**op_dct,random_state=self.random_state)
        clf.fit(self.data_sets['train_x'], self.data_sets['train_y'])

        try:
            train_r2 = regress_results(self.data_sets['train_x'], self.data_sets['train_y'], clf)[0].iloc[0, 1]
        except:
            train_r2 = None
        try:
            test_r2 = regress_results(self.data_sets['test_x'], self.data_sets['test_y'], clf)[0].iloc[0, 1]
        except:
            test_r2 = None
        try:
            future_r2 = regress_results(self.data_sets['data_future'], self.data_sets['target_future'], clf)[0].iloc[0, 1]
        except:
            future_r2 = None

        logger.info('current_params: \n{}'.format(op_dct))
        logger.info('train_ks: {}'.format(train_r2))
        logger.info('test_ks: {}'.format(test_r2))
        logger.info('future_ks: {}'.format(future_r2))

        if self.regress_flag == 'test':
            return 1 - test_r2
        else:
            return 1 - future_r2

    def regress_opt(self,n_trials=100):
        study = optuna.create_study()  # Create a new study.
        study.optimize(self.regress_objective, n_trials=n_trials)
        best_paras=study.best_params
        best_clf=self.model(**best_paras, random_state=self.random_state)
        best_clf.fit(self.data_sets['train_x'], self.data_sets['train_y'])
        try:
            train_r2 = regress_results(self.data_sets['train_x'], self.data_sets['train_y'], best_clf)[0].iloc[0, 1]
        except:
            train_r2 = None
        try:
            test_r2 = regress_results(self.data_sets['test_x'], self.data_sets['test_y'], best_clf)[0].iloc[0, 1]
        except:
            test_r2 = None
        try:
            future_r2 = regress_results(self.data_sets['data_future'], self.data_sets['target_future'], best_clf)[0].iloc[0, 1]
        except:
            future_r2 = None

        best_results = {'train_r2': train_r2,'test_r2': test_r2, 'future_r2': future_r2}

        return best_paras, best_clf, best_results


class grid_best:
    def __init__(self,space,model,cv):
        self.space=space
        self.model=model
        self.cv=cv

    def best_params(self,train_x,train_y):
        grid=GridSearchCV(self.model,self.space,cv=self.cv)
        grid.fit(train_x,train_y)
        best_params=grid.best_estimator_
        return best_params

class random_best:
    def __init__(self,space,model,cv,n_iter_search ):
        self.space=space
        self.model=model
        self.cv=cv
        self.n_iter_search=n_iter_search

    def best_params(self,train_x,train_y):
        random_search=RandomizedSearchCV(self.model,self.space,self.n_iter_search,cv=self.cv,iid=False)
        random_search.fit(train_x,train_y)
        best_params=random_search.best_params_
        return best_params


