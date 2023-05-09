import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
import hyperopt
import pickle

## Model Interface, support fit, predict, hyper-param, score
class resusemodel:
    def __init__(self, param):
        pass
    def fit(self,X,Y):  ## X to be dataframe or numpy array
        pass 
    def predict(self,X):
        pass
    def score(self, X,Y):
        pass

from sklearn.model_selection import KFold
from multiprocessing.pool import ThreadPool
import time

class CrossValidation:
    
    def __init__(self, modelbuilder, datax, datay , logfile , cvparam = {'num_folds':10, 'shuffle': True, 'random_state':17}):
        self.param = cvparam
        self.mdbulder = modelbuilder
        self.folds = KFold(n_splits= cvparam['num_folds'],shuffle = cvparam['shuffle'], random_state = cvparam['random_state'])
        self.datax = datax
        self.datay = datay
        self.log = open(logfile,"w")

    def evaluate(self, param, multithread = True):
        score = []
        t1 = time.time()
        if not multithread:
            
            for n_fold, (train_idx, valid_idx) in enumerate(self.folds.split(self.datax, self.datay)):
                train_x, train_y = self.datax.iloc[train_idx], self.datay.iloc[train_idx]
                valid_x, valid_y = self.datax.iloc[valid_idx], self.datay.iloc[valid_idx]
                model_fold = self.mdbulder(param)
                model_fold.fit(train_x, train_y)
                scorefold = model_fold.score(valid_x,valid_y)
                print('FOLD {}'.format(n_fold) + "score: " + str(scorefold))
                self.log.write('FOLD {}'.format(n_fold) + "score: " + str(scorefold) + "\n")
                self.log.flush()
                score.append(scorefold)
            print("Final CV score: "+ str(sum(score)/len(score)) + " for param" + str(param))
            self.log.write("Final CV score: "+ str(sum(score)/len(score)) + " for param" + str(param) + "\n")
            t2 = time.time()
            self.log.write(" Evaluation use %f seconds \n"%(t2 - t1))
            
            self.log.flush()
            return sum(score)/len(score), score
        else:
            splits = list(self.folds.split(self.datax, self.datay))
            def task(n):
                (train_idx, valid_idx) = splits[n]
                train_x, train_y = self.datax.iloc[train_idx], self.datay.iloc[train_idx]
                valid_x, valid_y = self.datax.iloc[valid_idx], self.datay.iloc[valid_idx]
                model_fold = self.mdbulder(param)
                model_fold.fit(train_x, train_y)
                scorefold = model_fold.score(valid_x,valid_y)
                print('FOLD {}'.format(n) + "score: " + str(scorefold))
                self.log.write('FOLD {}'.format(n) + "score: " + str(scorefold) + "\n")
                self.log.flush()
                return scorefold
            with ThreadPool(5) as pool:
        # call the same function with different data concurrently
                for result in pool.map(task, range(10)):
                   score.append(result)
            print("Final CV score: "+ str(sum(score)/len(score)) + " for param" + str(param))
            self.log.write("Final CV score: "+ str(sum(score)/len(score)) + " for param" + str(param) + "\n")
            t2 = time.time()
            self.log.write(" Evaluation use %f seconds \n"%(t2 - t1))    
            self.log.flush()
            return sum(score)/len(score), score


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    
def OptimizeHSpace(datax, datay, modelclass, logfile ,hspace,cvparam = {'num_folds':10, 'shuffle': True, 'random_state':17},
                   hpparam={'algo': tpe.suggest, 'max_evals':10}):
    trials = Trials()
    cvevaluator = CrossValidation(modelclass, datax, datay,logfile, cvparam)
    cvdetails = []
    def f(param):
        rtn = cvevaluator.evaluate(param)
        cvdetails.append(rtn[1])
        return rtn[0]
    best = fmin(fn=f, space=hspace, algo=hpparam['algo'], max_evals=hpparam['max_evals'], trials=trials)
    return best, trials, cvdetails
                    
    

### LightGBM 
def dfto32(df,to32types = [np.float64, np.int64]):
    convertlist = []
    for tp in to32types:
        convertlist = convertlist + list(df.columns[df.dtypes ==tp])
    for i in convertlist:
        df[i] = df[i].astype(np.float32)
    return df

def lightgbmdfprocess(df,featurecols, tgtcol ,dropyna=True, rtntgtcol = True, to32 = True):
    if dropyna:
        df = df[~df[tgtcol].isna()]
    if to32:
        df = dfto32(df)
    if rtntgtcol:
        return df[featurecols],df[tgtcol]
    else:
        return df[featurecols]


def cleanspaceindict(d):
    for x in d:
        if " " in d:
            d[x.strip(" ")] = d[x]
    return d


    
import math
class LightGBMModel:
    def __init__(self, param, defaultparam={'objective': 'regression','verbose': 1,'device':'gpu', 'metric':'rmse', 'num_thread' :1 }):
        self.param = param
        self.defaultparam = defaultparam.copy()
        self.defaultparam.update(param)
        self.defaultparam["num_leaves"] = self.defaultparam["num_leaves"]*100
        self.defaultparam = cleanspaceindict(self.defaultparam)
        self.model = None
        
    def fit(self,X,Y):
        traindata = lgb.Dataset(X,label=Y)
        paramtrain = self.defaultparam
        nround = self.defaultparam['nround']
        paramtrain.pop("nround",None)
        self.model = lgb.train(paramtrain, traindata,num_boost_round= nround)
    
    def refitdf(self ,df,feature,tgt):
        datax,datay = lightgbmdfprocess(df,feature,tgt)
        traindata = lgb.Dataset(datax,label=datay)
        paramtrain = self.defaultparam
        nround = self.defaultparam['nround']
        paramtrain.pop("nround",None)
        self.model = lgb.train(paramtrain, traindata,num_boost_round= nround)
        
        
    def predict(self, df, feature):
        return self.model.predict(df[feature])
    
    def score(self,X,Y):
        predy = self.model.predict(X)
        error = predy - Y.values
        return math.sqrt((error*error).mean())
    
    def getdescription():
        return "LGB Model L2 penalty"

def splitfuc(df,splitcol, splitval):
    return df[df[splitcol]<splitval], df[df[splitcol]>=splitval]

def simulatesimplepportfolio(df,rtncol,predcol,qtl,weighted=False):
    qtls = np.quantile(df[predcol],[qtl,1.0 - qtl])
    dfs = df[(df[predcol]<= qtls[0]) | (df[predcol]>=qtls[1])]
    if weighted:
        return (dfs[rtncol]*dfs[predcol]).sum()/ (dfs[predcol].abs().sum())
    else:
        return ((-df[df[predcol]<= qtls[0]][rtncol].mean()) + df[df[predcol]>= qtls[1]][rtncol].mean())*0.5
### Help us to compare the different models  ~~~ cor, rtn , avg rtn?
def evaluate(df ,  rtncol="midrtn55",predcol= "predy" ):
    m1 = df[[rtncol,predcol]].corr().values[0][1]
    m2 = df.groupby("date").apply(lambda x: x[[rtncol,predcol]].corr().values[0][1]).describe()
    m3 = df.groupby("date").apply(lambda x: (((x[rtncol] - x[rtncol].mean())*x[predcol]).sum())/ (x[predcol].abs().sum())).describe()
    m4 = df.groupby("date").apply(lambda x:simulatesimplepportfolio(x,rtncol,predcol,0.05 )).describe()
    m5 = df.groupby("date").apply(lambda x:simulatesimplepportfolio(x,rtncol,predcol,0.05, True )).describe()
    return [m1,m2,m3,m4,m5]

def enlistdic(x):
    r = x.copy()
    for a in r:
        r[a] = [r[a]]
    return r

def applyintmap(d,intcol):
    for c in intcol:
        d[c] = int(d[c])
    return d
    
import time
import pandas as pd
import os

import math


class OAExperiment:
    projectname = "AuctionSignal"
    rootpath = "/datassd/datastore/AuctionSignal/"
    recordtable = "Experiments.csv"
    
    def __init__(self, datapath, model, traintestsplit,splitdate, space, maxeval,cvparams, comment, description ,experimentname = None):
        self.model = model
        self.datapath = datapath
        self.split = traintestsplit
        self.splitdate = splitdate
        self.comment = comment
        self.expid = int(time.time())
        self.maxeval = maxeval
        self.cvparams = cvparams
        self.space = space
        self.experimentname = experimentname if not experimentname is None else str(self.expid)
        self.logname = self.experimentname + ".log"
        self.description = description
        
    
    def runfitting(self):
        dataf = open(  self.datapath ,"rb")
        data = pickle.load(dataf)
        self.dftrain,  self.dftest =  self.split(data[0])
        self.feature = data[2]
        self.tgt = data[1]
        ### not good to put it here, let's move to model
        datax,datay = lightgbmdfprocess(self.dftrain, self.feature, self.tgt)
        rtn = OptimizeHSpace(datax,datay, self.model, self.logname ,self.space ,hpparam={'algo': tpe.suggest, 'max_evals':self.maxeval}, cvparam = self.cvparams )
        return list(rtn)
    
    def getFullTests(self, param, keepcol = ["date","symbol"]):
        dftrain = self.dftrain
        dftest = self.dftest
        feature = self.feature
        tgt = self.tgt
        mopt = self.model(param)
        mopt.refitdf(dftrain,feature,tgt)
        y = mopt.predict(dftest,feature)
        err = dftest[tgt] - y
        testerr = math.sqrt((err*err).mean())
        testpred = dftest[keepcol + [tgt]]
        testpred["predy"] = y
    
    ### cross validate training
        sfolds = KFold(n_splits= 10, shuffle = False)
        cvlist = []
        for n_fold, (train_idx, valid_idx) in enumerate(sfolds.split(dftrain)):
                print("on %d fold"%n_fold)
                train_x = dftrain.iloc[train_idx]
                valid_x = dftrain.iloc[valid_idx]
                mopt = self.model(param)
                mopt.refitdf(train_x,feature,tgt)
                y = mopt.predict(valid_x,feature)
                cvinfo = valid_x[keepcol + [tgt]]
                cvinfo["predy"] = y
                cvlist.append(cvinfo)
        rtnx = pd.concat(cvlist)
        err2 = rtnx[tgt] - rtnx["predy"]
        cvtrainerr = math.sqrt((err2*err2).mean())    
        evaltest = evaluate(testpred)
        
        return mopt, testerr,cvtrainerr,evaltest, rtnx, testpred
     
    def getTestMeasurs(testerr, cvtrainerr, evaltest):
        m1,m2,m3,m4,m5 = evaltest
        rtn = dict()
        rtn["TestRMSE"] = testerr
        rtn["TestCor"] = m1
        rtn["TestCorByDay"] = (m2["mean"], m2["std"])
        rtn["TestWavgRtn"] = (m3["mean"], m3["std"])
        rtn["TestQtlAvgRtn"] = (m4["mean"], m4["std"])
        rtn["TestQtlWavgRtn"] = (m5["mean"], m5["std"])
        rtn["TsCVRMSE"] = cvtrainerr
        return rtn
        
    
    ## Result Table -- ExId | DataPath | DataDescription | Model Description | Test/Train Split | CV details | KeyFittedResult | KeyTestResults 
    def setrecord(self,optresults, testerr,cvtrainerr,evaltest):
        try:
            origdf = pd.read_csv(OAExperiment.rootpath + OAExperiment.recordtable,sep="|")
        except Exception:
            origdf = pd.DataFrame()
        rtncol = dict()
        rtncol["ExpId"] = self.expid
        rtncol["DataPath"] = self.datapath
        rtncol["DataDescription"] =  self.description["DataDescription"]
        rtncol["ModelDescription"] = self.model.getdescription()
        rtncol["TrainTestSplit"] = self.splitdate
        rtncol.update(OAExperiment.getTestMeasurs(testerr, cvtrainerr, evaltest))
        rtncol["FitResult"] = str(optresults[0])
        err = np.array([sum(x)/1.0 for x in optresults[2]])
        rtncol["ExpErrRange"] = (err.mean(),err.std())
        rtncol["CV"] = str(self.cvparams)
        rtncol["Status"] = "Normal"
        rtncol["Comment"] = self.comment
        rtncoldf = pd.DataFrame(enlistdic(rtncol))
        newdf = pd.concat([origdf, rtncoldf])
        newdf.to_csv(OAExperiment.rootpath + OAExperiment.recordtable, sep="|", index = False)
        return rtncol
        
    ## Dumpable result --  Test results, Model Parameters, Fitted Model, Residual DB1,2, Reproducible results, Importance~
    def dumpresult(self,  optrtn, selectedmodel, testeval, residualtrain, residualtest):
        dumpfolder = OAExperiment.rootpath + self.experimentname
        try:
            ##makedir
            os.mkdir(dumpfolder)
        except Exception:
            pass
        #Dump Fitting Parameters
        with open(dumpfolder + "/" + "OptRtn", 'wb') as f:
            pickle.dump(optrtn, f)
        
        with open(dumpfolder + "/" + "optparams", 'w') as f:
            f.write(str(optrtn[0]))
        
        #Dumpt Fitted Model
        with open(dumpfolder + "/" + "SelectedModel", 'wb') as f:
            pickle.dump(selectedmodel, f)
        #Evaluation Result
        with open(dumpfolder + "/" + "Evaluation", 'wb') as f:
            pickle.dump(testeval, f)
        ## some visible stuff
        imp = pd.DataFrame( { "feature": selectedmodel.model.feature_name()  ,"importance": selectedmodel.model.feature_importance()})
        imp.to_csv(dumpfolder + "/featureimportance.csv")
        lgb.plot_importance(selectedmodel.model,figsize=(12,12)).figure.savefig(dumpfolder + "/importance.png")
        
        ### Dump reproducable numbers, informations, 
        residualtrain.to_csv(dumpfolder + "/" + "trainresidual.csv")
        residualtest.to_csv(dumpfolder + "/" + "testresidual.csv")

        
 