# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:02:39 2018

@author: Mattia Bolzoni
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:53:54 2018

@author: matti
"""
###packages    
import tensorflow as tf
import scipy
import numpy as np
import pandas as pd        
import datetime
import os
import matplotlib.pyplot as plt

testing=False
###Parameters

n_l1 = 100 #hidden nodes of layer 1
n_l2 = 100 #hidden nodes of layer 2
batch_size = 128
n_epochs = 5
learning_rate = 0.001
beta1 = 0.9
results_path = './Results'#where to save results and logs
delay=22#how many days of the past of the title to use as regressors for allocation
time_window=1 #how many days in the future the allocation will be hold
select_titoli=range(0,10)#which columns of the datafile are considered tradable assets
N_rows=None#how many rows use from the datafile
ptf_dim = len(select_titoli)
train_perc=0.5#the percentage of the data putted in the training data
test_perc=0.49#the percentage of the data putted in the test data
commissions=0.01
file_data_path=r'C:\Bolz\Dati\DatiIndexesEquityDaFoglioFabrizio.csv'

def form_results(p_cv_round):
    """
    Forms folders for each run to store the tensorboard files, saved models, log files and figures.
    :param p_cv_round: the cross-validation round to put in the folder name
    :return: four strings pointing to tensorboard, saved models, log paths and figures path respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_nnetPtf_cv_{4}". \
        format(datetime.date.today(), datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second,p_cv_round)
    folder_path=results_path + folder_name 
    tensorboard_path = folder_path+ '/Tensorboard'
    saved_model_path = folder_path+ '/Saved_models/'    
    log_path=folder_path+'/log.txt'
    img_path_local=folder_path
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)        
    return tensorboard_path, saved_model_path, log_path,img_path_local


def plot_results(documents,img_path):
    """
    Plot the results of a single run, prints also some statistics in the prompt. Saves the image in the given path.
    :param documents: dictionary containing as values dictionaries with the sharpe-ratio and volatility for each strategy
    :param img_path: the path to save the figure
    :return: None
    """
    
    document_nnet=documents['nnet']
    document_nnet_comm=documents['nnet_comm']
    document_nnet_freeze=documents['nnet_freeze']
    document_ew=documents['ew']
    document_minvola=documents['minvola']
    document_rsk_par=documents['rsk_par']
    document_momentum=documents['momentum']
    
    plt.plot('vola', 'sharpe', data=document_nnet,color='black')
    plt.plot('vola', 'sharpe', data=document_nnet_comm,color='orange')
    plt.plot('vola', 'sharpe', data=document_nnet_freeze,color='blue')
    plt.plot('vola', 'sharpe', data=document_ew, color='green', marker='o')
    plt.plot('vola', 'sharpe', data=document_minvola, color='red', marker='s')
    plt.plot('vola', 'sharpe', data=document_rsk_par, color='purple', marker='^')
    plt.plot('vola', 'sharpe', data=document_momentum, color='yellow', marker='H')
    
    plt.legend(('nnet', 'nnet+comm', 'nnet freezed', 'nnet freezed', 'minvola', 'equal risk', 'momentum'))        
    
    print("ew")                
    print(document_ew)                
    print("min vola")
    print(document_minvola)
    print("risk parity")
    print(document_rsk_par)
    print("nnet")
    print(document_nnet['sharpe'])
    plt.savefig(img_path + "/sharpe_vs_vola.png")

def plot_results_2_4_row(documents_cv_all,img_path):
    """
    Plot the results of a multiple run on a single facet, prints also some statistics in the prompt. Saves the image in the given path.
    :param documents: dictionary containing, for each cross-validated round a dictionary with values dictionaries with the sharpe-ratio and volatility for each strategy
    :param img_path: the path to save the figure
    :return: None
    """
    righe = int(len(documents_cv_all)/2)
    fig,axes = plt.subplots(righe, 2, sharex=True, sharey=True)
    for cv in documents_cv_all:
        document_nnet=documents_cv_all[cv]['nnet']
        document_nnet_comm=documents_cv_all[cv]['nnet_comm']
        document_nnet_freeze=documents_cv_all[cv]['nnet_freeze']
        document_ew=documents_cv_all[cv]['ew']
        document_minvola=documents_cv_all[cv]['minvola']
        document_rsk_par=documents_cv_all[cv]['rsk_par']
        document_momentum=documents_cv_all[cv]['momentum']
        
        colonna=int(cv % 2)
        riga=int((cv-colonna)/2)
        myplt=axes[riga,colonna]
        myplt.plot('vola', 'sharpe', data=document_nnet,linestyle='-',color='black')
        myplt.plot('vola', 'sharpe', data=document_nnet_comm,linestyle='-.',color='orange')
        myplt.plot('vola', 'sharpe', data=document_nnet_freeze,linestyle=':',color='blue')
        myplt.plot('vola', 'sharpe', data=document_ew, color='green', marker='o')
        myplt.plot('vola', 'sharpe', data=document_minvola, color='red', marker='s')
        myplt.plot('vola', 'sharpe', data=document_rsk_par, color='purple', marker='^')
        myplt.plot('vola', 'sharpe', data=document_momentum, color='purple', marker='H')
        
        
        
        
        print("ew")                
        print(document_ew)                
        print("min vola")
        print(document_minvola)
        print("risk parity")
        print(document_rsk_par)
        print("momentum")
        print(document_momentum)
        print("nnet")
        print(document_nnet['sharpe'])
    
    fig.savefig(img_path + "/sharpe_vs_vola_all_cv.png")    
    fig.show()


def load_data(file_source,tw,delay,select_titoli=None,N_rows=None,cv_split_time=True):    
    """
    Load the returns from the given file, importing only the give columns and the given rows. Dividing them into future returns and past data. Divide it also in train data and test data.
    :param file_source: the path of the file with data. Is None provided, random gaussian data are used (for testing).
    :param tw: the future time window, aka how many days in the future the allocation will be hold
    :param delay: how many days of the past of the title to use as regressors for allocation    
    :param select_titoli: which columns to be considered tradable assets. If None, then all columns will be used.
    :param N_rows: how many rows to be imported from data. If None, then all will be used.
    :param cv_split_time: If True then every observations not in the training data goes in the test, otherwise just makes a sequential split between train and test
    :return: The name of the tradable assets, the regressors of train data, the future returns of train data, the regressors of test data, the future returns of test data, the number of tradable assets
    """
    if not file_source:        
        dati=np.random.randn(N_rows,select_titoli.shape[0])        
        df=dati
    else:
        dati = pd.DataFrame.from_csv(file_source, sep=',')     
        if not select_titoli:
            select_titoli= range(dati.shape[1])
        if not N_rows:    
            N_rows=dati.shape[0]        
        dati=dati.iloc[0:N_rows,select_titoli]#pandas dataframe
        titles_name=dati.columns.values        
        df=np.log(dati) - np.log(dati.shift(1)) #numpy returns matrix
        df=df.as_matrix()[1:df.shape[0],:]
    n_titles=len(select_titoli)    
    total_len_cutted=(df.shape[0]-delay-2*tw)
    if cv_split_time:
        first_train_time=tw+np.random.choice(int(total_len_cutted*(1-train_perc)),size=1)[0]
    else:
        first_train_time=np.random.choice(int(total_len_cutted*(1-train_perc-test_perc)),size=1)[0]

    first_train_time=np.max([first_train_time,tw])
    last_train_time=first_train_time+int(total_len_cutted*train_perc)        
    first_test_time=last_train_time+tw+1
    last_test_time=df.shape[0]-tw-delay    
    
    df_delay=np.ndarray([total_len_cutted,n_titles*delay],np.float32)
    df_future=np.ndarray([total_len_cutted,n_titles],np.float32)
    df_future[:]=0
    df_1delay=np.ndarray([total_len_cutted,n_titles],np.float32)    
    indice_cutted_time=list (range(total_len_cutted))    
    indice_cutted_time=[x+delay for x in indice_cutted_time] 
    df_1delay=np.copy(df[indice_cutted_time,:])
    indice_past=np.copy(indice_cutted_time)
    indice_future=np.copy(indice_cutted_time)
    for lag in range(delay):                    
        for titolo in range(n_titles):
            col=(titolo*delay)+lag
            df_delay[:,col]=np.copy(df[indice_past,titolo])
        indice_past=[x-1 for x in indice_past]            
    
    for lag in range(tw):        
        indice_future=[x+1 for x in indice_future] 
        df_future[:,:]=df_future[:,:]+np.copy(df[indice_future,0:n_titles]  )       
    
    if cv_split_time:
        x_train=df_delay[first_train_time:last_train_time,:]
        x_test=np.concatenate([df_delay[tw:(first_train_time-tw),:], df_delay[first_test_time:last_test_time,:]],axis=0)
        returns_test=np.concatenate([df_future[tw:(first_train_time-tw),:], df_future[first_test_time:last_test_time,:]],axis=0)#*np.repeat(rescaling,last_test_time-first_test_time,axis=0)
    else:
        x_train=df_delay[first_train_time:last_train_time,:]
        x_test=df_delay[first_test_time:last_test_time,:]
        returns_test=df_future[first_test_time:last_test_time,:]
    
    returns_train=df_future[first_train_time:last_train_time,:]
    
    return titles_name,x_train,returns_train,x_test,returns_test,n_titles


def dati_get_next_batch_new(batch_size,data_train,returns_train,cursor):                   
    """
    Extract the next batch to train via SGD from the data
    :param batch_size: how many observations to put in the batch
    :param data_train: the data to extract from.
    :param data_train: the future returns to extract from
    :param cursor: used to memorize the starting point from the extraction (used to deal with sequential batches)
    :return: The batch of the regressors, the batch of the returns, the cursor that memorize the position
    """

    global permutations_sgd
    randomize=False
    if randomize:
        indice= np.random.choice((data_train.shape[0]),size=batch_size,replace=True) #.randperm(x.size(0))
    else:
        if (not cursor):
            cursor=0
        if (cursor==0) or (cursor+batch_size)>data_train.shape[0]:
            cursor=0    
            permutations_sgd= np.random.choice((data_train.shape[0]),size=data_train.shape[0],replace=False) #.randperm(x.size(0))
        indice=permutations_sgd[list(range(cursor,cursor+batch_size)        )]
        cursor=cursor+batch_size
    batch_x=data_train[indice,:]
    future_returns=returns_train[indice,:]
    return batch_x, future_returns, cursor

###Portfolio strategies
def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out

#The ptf manager
def ptf_allocator(x, name,ptf_dim,reuse=False):
    """    
    :param x: input to the ptf manager
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which are the weights of the titles in the ptf.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope(name):
        e_dense_1 = tf.nn.relu(dense(x, x.shape[1], n_l1, 'e_dense_1_{}'.format(name)))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2_{}'.format(name)))
        e_w = tf.nn.softmax(dense(e_dense_2, n_l2, ptf_dim, 'e_weights_nonnorm_{}'.format(name)))
        return e_w

def from_weights2sharpe(w,r,commission_perc):
    """    
    :param w: weights of the ptf
    :param r: returns of the future titles
    :param commission_perc: the commissions in percentage.
    :return: the sharpe ratio of a ptf with the given weights. Is dynamically rebalanced (sums the log-returns).
    """
    ret_ptf=tf.reduce_sum(tf.multiply(w,r),axis=1)
    commissions=tf.reduce_mean(tf.abs(w[0:(w.shape[0]-1),:]-w[1:(w.shape[0]),:]))*commission_perc
    mean_ret_ptf=tf.reduce_mean(ret_ptf)
    sd_ret_ptf=tf.sqrt(tf.reduce_mean(tf.square(ret_ptf))-tf.square(mean_ret_ptf))
    ptf_sharpe=tf.divide(tf.subtract(mean_ret_ptf,commissions),sd_ret_ptf)
    return ptf_sharpe,sd_ret_ptf


def ptf_risk_parity_perf(data_train,batch_future_returns2test,w0=None):
    """    
    :param data_train: the titles returns used to estimate the risk parity weights.
    :param future_returns: returns of the future titles to test the ptf performances.
    :param wo: starting weights for the estimation. If None, the risk budgets for each title is used as weight.
    :return: the estimated weights, the Sharpe-ratio of the ptf and the volatility of the ptf
    """
    from scipy.optimize import minimize
    def assets_meanvar(data_train):        
        E_ret=data_train.mean(axis=0)
        Vcv=np.cov(data_train.T)
        return E_ret, Vcv
    # Estimate assets's expected return and covariances
    E_ret, V = assets_meanvar(data_train)
         # risk budgeting optimization
    def calculate_portfolio_var(w,V):
        # function that calculates portfolio risk
        w = np.matrix(w)
        return (w*V*w.T)[0,0]
    
    def calculate_risk_contribution(w,V):
        # function that calculates asset contribution to total risk
        w = np.matrix(w)
        sigma = np.sqrt(calculate_portfolio_var(w,V))
        # Marginal Risk Contribution
        MRC = V*w.T
        # Risk Contribution
        RC = np.multiply(MRC,w.T)/sigma
#        RC=MRC/sigma
        return RC
    
    def risk_budget_objective(x,pars):
        # calculate portfolio risk
        V = pars[0]# covariance table
        x_t = pars[1] # risk target in percent of portfolio risk
        sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p,x_t))
        asset_RC = calculate_risk_contribution(x,V)
        J = 1000*sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error, the *1000 is used to increase numerical efficacy
        return J
    
    def total_weight_constraint(x):
        return np.sum(x)-1.0
    
    def long_only_constraint(x):
        return x
    
    dim_weights= data_train.shape[1]
    x_t = np.full(dim_weights,1/dim_weights).T # your risk budget percent of total portfolio risk (equal risk)
    if w0.any()==None:
        w0=np.copy(x_t)
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
    {'type': 'ineq', 'fun': long_only_constraint})
    res= minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP',constraints=cons)#, options={'disp': True} #add the commented parameters to have verbosity about the minimization
    weights = res.x    
    print("weights riskPar: {}".format(str(weights)))    
    log_ret_rsk_par_test=np.dot(batch_future_returns2test,weights)
    vola_rsk_par_test=log_ret_rsk_par_test.std()
    sharpe_rsk_par_test=log_ret_rsk_par_test.mean()/vola_rsk_par_test
    print("sharpe riskPar: {}".format(str(sharpe_rsk_par_test)))    
    return weights,sharpe_rsk_par_test,vola_rsk_par_test
    


def ptf_min_vola_perf(data_train,future_returns):
    """    
    :param data_train: the titles returns used to estimate the risk parity weights.
    :param future_returns: returns of the future titles to test the ptf performances.
    :return: the estimated weights, the Sharpe-ratio of the ptf and the volatility of the ptf
    """    
    def assets_meanvar(data_train):
        E_ret=data_train.mean(axis=0)
        Vcv=np.cov(data_train.T)
        return E_ret, Vcv
    # Estimate assets's expected return and covariances
    E_ret, Vcv = assets_meanvar(data_train)
    rf = 0   # Define Risk-free rate
    
    # Calculates portfolio mean return
    def port_mean(W, R):
        return sum(R * W)
    
    # Calculates portfolio variance of returns
    def port_var(W, C):
        return np.dot(np.dot(W, C), W)
    
    # Combination of the two functions above - mean and variance of returns calculation
    def port_mean_var(W, R, C):
        return port_mean(W, R), port_var(W, C)
    # Given risk-free rate, assets returns and covariances, this function calculates
    # weights of tangency portfolio with respect to sharpe ratio maximization
    def solve_weights(R, C, rf):
        def fitness(W, R, C, rf):
            # calculate mean/variance of the portfolio
            mean, var = port_mean_var(W,R,C)  
            return 1000*np.sqrt(var) #minimize the variance, the *1000 is used to increase numerical efficacy
#            util = (mean - rf) / np.sqrt(var)      # utility = Sharpe ratio
#            return 1/util                       # maximize the utility
        n = len(R)
        W = np.ones([n])/n                     # start with equal weights
        b_ = [(0.,1.) for i in range(n)]    # weights between 0%..100%. 
                                            # No leverage, no shorting
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })   # Sum of weights = 100%    
        optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), 
                    method='SLSQP', constraints=c_, bounds=b_)  
        if not optimized.success: 
            raise BaseException(optimized.message)
        return optimized.x  # Return optimized weights
    
    weights=solve_weights(E_ret, Vcv, rf)
    print("weights minVola: {}".format(str(weights)))    
    log_ret_min_vola_test=np.dot(future_returns,weights)
    vola_min_vola_test=log_ret_min_vola_test.std()
    sharpe_min_vola_test=log_ret_min_vola_test.mean()/vola_min_vola_test
    print("sharpe minVola: {}".format(str(sharpe_min_vola_test)))    
    return weights,sharpe_min_vola_test,vola_min_vola_test
    
def ptf_momentum_perf(data_test,future_returns_test,n_titles,delay,n_titles2select):
    """
    :param data_test: the past returns of the titles (test data) used to estimate the momentum.
    :param future_returns: returns of the future titles to test the ptf performances.
    :return: the Sharpe-ratio of the ptf and the volatility of the ptf
    """
    momentum_perf=np.ndarray([future_returns_test.shape[0],n_titles])
    for titolo in range(n_titles):
        momentum_perf[:,titolo]=data_test[:,(titolo)*delay:(titolo+1)*delay].sum(axis=1)
    classifica_momentum=np.argsort(momentum_perf,axis=1)
    selected_momentum=np.ndarray([future_returns_test.shape[0],n_titles],bool)    
    selected_momentum[:]=False
    for giro in range(future_returns_test.shape[0]):
        selected_momentum[giro,classifica_momentum[giro,0:n_titles2select]]=True    
    returns_strategy_momentum=future_returns_test[selected_momentum].reshape(-1,n_titles2select).mean(axis=1)
    vola=returns_strategy_momentum.std()
    sharpe=returns_strategy_momentum.mean()/vola
    return sharpe,vola


def train(cv_round,p_cv_analysis,commissions):
    """
    Used to train the allocator by passing in the necessary inputs.
    :param cv_round: The number of cross-validation to use to create the save folder
    :param commissions: The commissions to be used
    :return: does not return anything
    """
    
    titles_name,data_train,future_returns_train,data_test,future_returns_test,n_titles= load_data(file_data_path,time_window,delay,select_titoli,N_rows,cv_split_time=True)       
    x_input_dim = data_train.shape[1]
    x_input2test=data_test.shape[0]
    #Placeholders:
    x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_input_dim], name='Input')
    x_allTrain = tf.placeholder(dtype=tf.float32, shape=[data_train.shape[0], x_input_dim], name='InputAllTrain')
    x_input2plot= tf.placeholder(dtype=tf.float32, shape=[x_input2test, x_input_dim], name='Input2plot')
    ptf_weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, ptf_dim], name='Ptf')    
    x_input_comm = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_input_dim], name='Input_comm')
    x_input2plot_comm = tf.placeholder(dtype=tf.float32, shape=[x_input2test, x_input_dim], name='Input2plot_comm')
    ptf_weights_comm = tf.placeholder(dtype=tf.float32, shape=[batch_size, ptf_dim], name='Ptf_comm')
    ptf2test = tf.placeholder(dtype=tf.float32, shape=[x_input2test, ptf_dim], name='Ptf_weights2plot')
    future_returns2test=tf.placeholder(dtype=tf.float32, shape=[x_input2test, ptf_dim], name='Titles_future_return')    
    future_returns2train=tf.placeholder(dtype=tf.float32, shape=[batch_size, ptf_dim], name='Titles_future_return')
    #Tensors:
    environment_train='train_params_cv_{}'.format(cv_round)
    environment_train_comm='train_params_cv_{}_comm'.format(cv_round)
    with tf.variable_scope(tf.get_variable_scope()):
        ptf_weights = ptf_allocator(x_input,environment_train,n_titles)
        ptf_weights_allTrain = ptf_allocator(x_allTrain,environment_train,n_titles,reuse=True)        
    with tf.variable_scope(tf.get_variable_scope()):            
        ptf_weights_comm = ptf_allocator(x_input_comm,environment_train_comm,n_titles)
    with tf.variable_scope(tf.get_variable_scope()):
        ptf2test  = ptf_allocator(x_input2plot,environment_train,n_titles,reuse=True)
        sharpe2test=from_weights2sharpe(ptf2test,future_returns2test,0)
        ptf2test_comm  = ptf_allocator(x_input2plot_comm,environment_train_comm,n_titles,reuse=True)
        sharpe2test_comm=from_weights2sharpe(ptf2test_comm,future_returns2test,commissions)    
    ptf_loss  =-(from_weights2sharpe(ptf_weights,future_returns2train,0)[0])
    ptf_loss_comm  =-(from_weights2sharpe(ptf_weights_comm,future_returns2train,commissions)[0])
    #optimizers:
    ptf_optimizer  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(ptf_loss )
    ptf_optimizer_comm  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(ptf_loss_comm )
    #init various:
    init = tf.global_variables_initializer()
    tf.summary.scalar(name='Ptf_Loss', tensor=ptf_loss)
    tf.summary.scalar(name='Ptf_Loss_comm', tensor=ptf_loss_comm)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()    
    step = 0
    #classic strategies:
    min_vola=True
    equal_weights=True
    risk_parity=True
    momentum_simple=True
    if min_vola:
        weights_minvola,sharpe_minvola,vola_minvola=ptf_min_vola_perf(future_returns_train,future_returns_test)    
    else: weights_minvola=None        
    if equal_weights:
        r_ew=future_returns_test.mean(axis=1)
        vola_ew=r_ew.std()
        sharpe_ew=r_ew.mean()/vola_ew        
        print("sharpe equal_weight: {}".format(sharpe_ew))    
    if risk_parity:
        weights_rsk_par,sharpe_rsk_par,vola_rsk_par=ptf_risk_parity_perf(future_returns_train,future_returns_test,weights_minvola)
    if momentum_simple:
        sharpe_momentum,vola_momentum=ptf_momentum_perf(data_test,future_returns_test,n_titles,delay,int(n_titles/2))
        
    
    document_ew={'sharpe':sharpe_ew,'vola':vola_ew}
    document_minvola={'sharpe':sharpe_minvola,'vola':vola_minvola}
    document_rsk_par={'sharpe':sharpe_rsk_par,'vola':vola_rsk_par}
    document_momentum={'sharpe':sharpe_momentum,'vola':vola_momentum}
    
    n_batches = int(data_train.shape[0] / batch_size)    
    
    sharpe_nnet=np.ndarray(n_batches*n_epochs)    
    vola_nnet=np.ndarray(n_batches*n_epochs)
    weights_nnet=np.ndarray([n_batches*n_epochs,n_titles])
    
    sharpe_nnet_comm=np.ndarray(n_batches*n_epochs)    
    vola_nnet_comm=np.ndarray(n_batches*n_epochs)
    weights_nnet_comm=np.ndarray([n_batches*n_epochs,n_titles])    
    
    sharpe_nnet_freeze=np.ndarray(n_batches*n_epochs)
    vola_nnet_freeze=np.ndarray(n_batches*n_epochs)
    weights_nnet_freeze=np.ndarray([n_batches*n_epochs,n_titles])
    #creates the saving folder
    tensorboard_path, saved_model_path, log_path, img_path = form_results(cv_round)
    with open(log_path, 'a') as log:
        template_log="Parameters:\nbatch_size: {}\nn_epochs: {}\nlearning_rate: {}\nbeta1: {}\ndelay: {}\ntime_window: {}\nfirst_title: {}\nn_titles: {}\nfile_source: {}\ncommissions: {}\n"
        log.write(template_log.format(batch_size,n_epochs,learning_rate,beta1,delay,time_window,select_titoli[0],n_titles,file_data_path,commissions))
                                
    counter_steps=0
    with tf.Session() as sess:        
        init = tf.global_variables_initializer()
        sess.run(init)        
        cursore=None
        if not p_cv_analysis:
            weights2plot_nnet={}
            weights2plot_nnet_comm={}
        for i in range(n_epochs):                
            print("------------------Epoch {}/{}------------------".format(i, n_epochs))                
            for b in range(1, n_batches + 1):
                batch_x,batch_future_returns,cursore = dati_get_next_batch_new( batch_size,data_train,future_returns_train,cursore)
                sess.run(ptf_optimizer  , feed_dict={x_input: batch_x,future_returns2train: batch_future_returns}) 
                sess.run(ptf_optimizer_comm  , feed_dict={x_input_comm: batch_x,future_returns2train: batch_future_returns})                 
                weights_test=sess.run(ptf2test,feed_dict={x_input2plot:data_test})
                sharpe_testing_itera, sd_testing_itera=sess.run(sharpe2test,feed_dict={x_input2plot:data_test,future_returns2test:future_returns_test})                
                weights_test_comm=sess.run(ptf2test_comm,feed_dict={x_input2plot_comm:data_test})
                sharpe_testing_itera_comm, sd_testing_itera_comm=sess.run(sharpe2test_comm,feed_dict={x_input2plot_comm:data_test,future_returns2test:future_returns_test})                
                weights_test_freeze=sess.run(ptf_weights_allTrain,feed_dict={x_allTrain:data_train })                                        
                ritorni_testing_freeze=np.dot(future_returns_test,weights_test_freeze.mean(axis=0))                    
                sd_testing_itera_freeze=ritorni_testing_freeze.std()
                sharpe_testing_itera_freeze=ritorni_testing_freeze.mean()/sd_testing_itera_freeze
                
                sharpe_nnet[counter_steps]=sharpe_testing_itera
                vola_nnet[counter_steps]=sd_testing_itera
                weights_nnet[counter_steps,:]=weights_test.mean(axis=0)                
                sharpe_nnet_comm[counter_steps]=sharpe_testing_itera_comm
                vola_nnet_comm[counter_steps]=sd_testing_itera_comm
                weights_nnet_comm[counter_steps,:]=weights_test_comm.mean(axis=0)
                sharpe_nnet_freeze[counter_steps]=sharpe_testing_itera_freeze
                vola_nnet_freeze[counter_steps]=sd_testing_itera_freeze
                weights_nnet_freeze[counter_steps,:]=weights_test_freeze.mean(axis=0)    
                
                if not p_cv_analysis:
                    weights2plot_nnet[counter_steps]=weights_test
                    weights2plot_nnet_comm[counter_steps]=weights_test_comm
                
                if b % 10 == 0:
                    print("Epoch: {}, iteration: {}".format(i, b))                        
                    print("media weights nnet: {}".format(weights_test.mean(axis=0)))
                    print("std weights nnet: {}".format(weights_test.std(axis=0)))
                    print("sharpe nnet: {}".format( sharpe_testing_itera))                  
                    with open(log_path, 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(i, b))
                        log.write("Sharpe nnet: {}\n".format(sharpe_testing_itera))
                        log.write("Vola nnet: {}\n".format(sd_testing_itera))
                        log.write("weights nnet: {}\n".format(weights_nnet[counter_steps,:]))
                counter_steps=counter_steps+1
                step += 1                
            document_nnet={'sharpe':sharpe_nnet[0:counter_steps],'vola':vola_nnet[0:counter_steps],'weights':weights_nnet[0:counter_steps,:]}                
            document_nnet_freeze={'sharpe':sharpe_nnet_freeze[0:counter_steps],'vola':vola_nnet_freeze[0:counter_steps],'weights':weights_nnet_freeze[0:counter_steps,:]}                
            document_nnet_comm={'sharpe':sharpe_nnet_comm[0:counter_steps],'vola':vola_nnet_comm[0:counter_steps],'weights':weights_nnet_comm[0:counter_steps,:]}                
        
        documents={'nnet':document_nnet,'nnet_comm':document_nnet_comm,'nnet_freeze':document_nnet_freeze,'ew':document_ew,'minvola':document_minvola,'rsk_par':document_rsk_par,'momentum':document_momentum}
        plot_results(documents,img_path)
        if not p_cv_analysis:               
            #the following outputs will change, depending on the type of output needed (not elegant, but it works)
            document_nnet=weights2plot_nnet
            document_nnet_comm=weights2plot_nnet_comm
            document_minvola=weights_minvola
            document_rsk_par=weights_rsk_par
            document_nnet_freeze=weights_nnet_freeze                
            def nnet_sampler(batch_x2analisi):
                weights_analisi=sess.run(ptf2test,feed_dict={x_input2plot:batch_x2analisi})
                return weights_analisi
        
            titolo_scelto=8 #which title will not have null synthetic returns (the others will)
            ana_positiva=np.ndarray(data_test.shape)
            ana_positiva[:]=0
            indici_singolo_titolo=np.ndarray(delay)
            for lag in range(delay):
                colonna=titolo_scelto+lag*n_titles
                indici_singolo_titolo[lag]=colonna
                ana_positiva[:,colonna]=np.abs(np.random.randn(ana_positiva.shape[0])*data_train[:,titolo_scelto].std())*3                
            weights_ana_positiva=nnet_sampler(ana_positiva)            
            ana_negativa=-ana_positiva            
            weights_ana_negativa=nnet_sampler(ana_negativa)            
            ana_neg_pos=np.copy(ana_negativa)            
            ana_neg_pos[:,int(ana_neg_pos.shape[1]/2):]=ana_positiva[:,int(ana_neg_pos.shape[1]/2):]            
            weights_ana_neg_pos=nnet_sampler(ana_neg_pos)            
            ana_pos_neg=np.copy(ana_negativa)            
            ana_pos_neg[:,:int(ana_neg_pos.shape[1]/2)]=ana_positiva[:,:int(ana_neg_pos.shape[1]/2)]            
            weights_ana_pos_neg=nnet_sampler(ana_pos_neg)            
            titolo2plot=titolo_scelto
            
            for titolo2plot in range(n_titles):
                fig,axes = plt.subplots(2, 2, sharex=True, sharey=True)            
                
                axes[0,0].hist(weights_ana_positiva[:,titolo2plot])
                plt.xlabel='pos-pos'
                axes[1,0].hist(weights_ana_neg_pos[:,titolo2plot])
                plt.xlabel='neg-pos'
                axes[0,1].hist(weights_ana_pos_neg[:,titolo2plot])
                plt.xlabel='pos-neg'
                axes[1,1].hist(weights_ana_negativa[:,titolo2plot])
                plt.xlabel='neg-neg'
                
                fig.savefig(img_path + "/weights_analisi_{}.png".format(titles_name[titolo2plot]))   
                
                plt.title=titolo2plot
            
            fig,axes = plt.subplots(2, 2, sharex=True, sharey=True)
            
            
            axes[0,0].hist(weights_ana_positiva[:,9:].sum(axis=1))
            axes[1,0].hist(weights_ana_neg_pos[:,9:].sum(axis=1))
            axes[0,1].hist(weights_ana_pos_neg[:,9:].sum(axis=1))
            axes[1,1].hist(weights_ana_negativa[:,9:].sum(axis=1))
            
            fig.savefig(img_path + "/weights_analisi_allCurr.png")   
            
            fig,axes = plt.subplots(2, 2, sharex=True, sharey=True)
            
            
            axes[0,0].hist(weights_ana_positiva[:,:10].sum(axis=1))
            axes[1,0].hist(weights_ana_neg_pos[:,:10].sum(axis=1))
            axes[0,1].hist(weights_ana_pos_neg[:,:10].sum(axis=1))
            axes[1,1].hist(weights_ana_negativa[:,:10].sum(axis=1))
            
            fig.savefig(img_path + "/weights_analisi_allEqty.png")   
            
            
            fig,axes = plt.subplots(2, 1, sharex=True, sharey=True)
            
            
            axes[0].hist(weights_test[:,9:].sum(axis=1))
            axes[1].hist(weights_test[:,:10].sum(axis=1))            
            
            fig.savefig(img_path + "/weights_test_curr_vs_eqty.png")   
            
            fig,axes = plt.subplots(2, 1, sharex=True, sharey=True)
            
            
            axes[0].hist(weights_test_comm[:,9:].sum(axis=1))
            axes[1].hist(weights_test_comm[:,:10].sum(axis=1))            
            
            fig.savefig(img_path + "/weights_test_curr_vs_eqty_comm.png")             
        else:               
            saver.save(sess, save_path=saved_model_path, global_step=step)                
        documents={'nnet':document_nnet,'nnet_comm':document_nnet_comm,'nnet_freeze':document_nnet_freeze,'ew':document_ew,'minvola':document_minvola,'rsk_par':document_rsk_par,'momentum':document_momentum}
        return documents,img_path,titles_name
            
if __name__ == '__main__':
    CV=4#CV=1 will do a single run and do also weights anaysis. CV odd or CV=2 will break, CV even and >2 will do a multiple cross-validation run
    documents_cv_all={}
    if CV>1:
        for cv_round in range(CV):        
            documents,img_path,titles_name=train(cv_round,True,commissions)
            if cv_round==0:
                img_path0=img_path#saves in the first folder the figure with every plot as a single facet
            documents_cv_all[cv_round]=documents
        plot_results_2_4_row(documents_cv_all,img_path0)
    else:
        documents,img_path,titles_name=train(0,False,commissions)
        document_nnet=documents['nnet']
        document_nnet_comm=documents['nnet_comm']
        document_nnet_freeze=documents['nnet_freeze']
        document_ew=documents['ew']
        document_minvola=documents['minvola']
        document_rsk_par=documents['rsk_par']
        document_momentum=documents['momentum']
        
        last_training_weights=document_nnet[len(document_nnet.keys())-1]
        last_training_weights_comm=document_nnet_comm[len(document_nnet_comm.keys())-1]
        
        percentiles_4_table=(10,50,90)
        cifre_arrotondamento=2
        extra_str_numero=3
        last_training_weights_perc=np.ndarray([len(select_titoli),len(percentiles_4_table)+extra_str_numero])
        last_training_weights_media=last_training_weights.mean(axis=0)
        last_training_weights_perc[:,0]=last_training_weights_media
        last_training_weights_perc[:,4]=document_minvola 
        last_training_weights_perc[:,5]=document_rsk_par
        
        last_training_weights_perc_comm=np.ndarray([len(select_titoli),len(percentiles_4_table)+1])
        last_training_weights_media_comm=last_training_weights.mean(axis=0)
        last_training_weights_perc_comm[:,0]=last_training_weights_media_comm
        
        ii=1
        for percentile in percentiles_4_table:
            last_training_weights_perc[:,ii]=np.percentile(last_training_weights,percentile,axis=0)
            last_training_weights_perc_comm[:,ii]=np.percentile(last_training_weights_comm,percentile,axis=0)
            ii+=1
        
        img_path0=img_path        
        
        
        stats4table=last_training_weights_perc
        stats4table_comm=last_training_weights_perc_comm
        
        table= pd.DataFrame(data=stats4table, index=titles_name, columns=('media',*percentiles_4_table,'minvola','equal risk'))
        table=round(table,cifre_arrotondamento)
        table_comm= pd.DataFrame(data=stats4table_comm, index=titles_name, columns=('media',*percentiles_4_table))
        table_comm=round(table_comm,cifre_arrotondamento)
        
        table.to_csv(img_path + r'\weights_stats_nnet.csv',sep='&',line_terminator='\\\\\n')
        table_comm.to_csv(img_path + r'\weights_stats_nnet_comm.csv',sep='&',line_terminator='\\\\\n')
        
