#!/usr/bin/env python
# coding: utf-8

# # The functions for tran_ncb, predict_ncb(using Decrete algo for continuous value)

# In[16]:


#calc conditional probility for continus attri
#1 bins
# take 2 files as input?
# have to use Laplace recorrect , next time
LABEL_LIST=["low", "median","high"]

def calc_condi_prob(X, y):
    # rows and cols
    m, n = X.shape
    # all possible y values.have 3 valus. 
    uni_values = y.unique()

    # output, 
    # have to know the range of y if init in this way 
    p_list=[]
    type_list=[]
    #prob of low, high, median
    p_prior=[]
    # print("m",m)
    # for y_ in uni_values:
    #     p_prior.append(  y[y==y_].value_counts()/m)
    TOTAL_LEBEL=3
    # LABEL_LIST=["low", "median","high"]
    for y_ in uni_values:        
        p_prior.append ((len(y[y == y_]) + 1) / (m + TOTAL_LEBEL) )

    # attrubute_nums = []
    # k=0
    
    for y_ in uni_values:
        X_ = X[y == y_]
        # the total num of this class
        total_num =X_.shape[0]
        p_label=[]
        for i in range(n):
            # is continus attri? 
            X_nonclassified = X.iloc[:, i]
            Xi = X_.iloc[:, i]

   
            type_list.append(type_of_target(Xi))
            if type_of_target(Xi) == 'continuous':
                # get the min and max of Xi, then divide to bins, then calc the num of every bins 

                NUM_BINS=10
                cast_xi = pd.cut(Xi,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                # p for (high|ABCD)
                # not m , should be the total num of this attri
                p= (cast_xi.value_counts()+1)/(total_num+NUM_BINS)
                # p_list[k].append(p)
                p_label.append(p)
            else  :
                # special handle for month , because 6,7 is missing
                if i == 1:
                    TOTAL_MON=12
                    Num_attris = TOTAL_MON
                else :
                    Num_attris = X_nonclassified.nunique()
                ret_list = []

                Xi_list = Xi.tolist()
                
                xi_min = np.min(X_nonclassified)
                # correct
                # print("xi_min", "i_", i, xi_min)
                STEP=1
                # [start, end, step]
                # for j in range(Num_attris):
                for j in range(xi_min, xi_min+Num_attris, STEP):
                    num_attri = Xi_list.count(j)
                    # print("X_nonclassified",X_nonclassified)
                    # print("i_",i, "j_",j, "count_", num_attri,"Num_attris", Num_attris )
                    p = (num_attri+1)/(total_num+Num_attris)
                    ret_list.append(p)
                # how handle output， list to series
                # but index  is consitant or not?, 
                # default index is 0,1,2,...
                # assign the index start point
                # p_list[k].append(pd.Series(ret_list, index =list(range(xi_min, xi_min+Num_attris, STEP))))
                p_label.append(pd.Series(ret_list, index =list(range(xi_min, xi_min+Num_attris, STEP))))                   
        
        # k=k+1
        p_list.append(p_label)
        # p0 , p
    return p_prior, p_list, type_list


# assume global value: num_prior =3
# assume global the name of this prior/label {low, median , high}
def predict_eval(p_prior, p_list, X, y):
    m, n = X.shape
    # ret=[]
    acc=0
    for i in range(m):
        # 'numpy.float64' object cannot be interpreted as an integer
        Xi=X.iloc[i, :].astype(object)
      
        perdict = predict_single(p_prior, p_list, Xi)
        if(perdict==y[i]):
            acc+=1 
    acc_p = acc/len(X)       
    # print("Acc :",acc/len(X))
    return acc_p
    # output label
def predict_single(p_prior, p_list, X):
    n = X.shape[0] #len(X)
    # X.shape 1, n 
    # 10
    # print(n)
    x_p0 ,x_p1, x_p2=0,0,0

    x_list=[x_p0, x_p1,x_p2]
    NUM_P_LIST=3
    for i in range(NUM_P_LIST):
        # err
        x_list[i] =np.log(p_prior[i]) 
        for j in range(n):

            # 3*10
            x_list[i]+=np.log(p_list[i][j][X[j]])
            # print( j,"_ ", X[j])
        x_list[i]=x_list[i].tolist()
    return LABEL_LIST[x_list.index(np.max(x_list))]





# # The functions for tran_ncb, predict_ncb(GAUSSIAN)
# UPDATE : 2020/11/20

# In[17]:


def calc_condi_prob_gaussian(X, y):
    # rows and cols
    m, n = X.shape
    # all possible y values.have 3 valus. 
    uni_values = y.unique()

    # output, 
    # have to know the range of y if init in this way 
    # p0_list=[]
    # p1_list=[]
    # p2_list=[]
    # p_list=[p0_list, p1_list, p2_list]
    p_list=[]
    type_list=[]
    #prob of low, high, median
    p_prior=[]
    # print("m",m)
    # for y_ in uni_values:
    #     p_prior.append(  y[y==y_].value_counts()/m)
    TOTAL_LEBEL=3
    # LABEL_LIST=["low", "median","high"]
    for y_ in uni_values:        
        p_prior.append ((len(y[y == y_]) + 1) / (m + TOTAL_LEBEL) )

    # attrubute_nums = []
    # k=0
    
    for y_ in uni_values:
        X_ = X[y == y_]
        # the total num of this class
        total_num =X_.shape[0]
        p_label=[]
        for i in range(n):
            # is continus attri? 
            X_nonclassified = X.iloc[:, i]
            Xi = X_.iloc[:, i]

   
            type_list.append(type_of_target(Xi))
            if type_of_target(Xi) == 'continuous':
             
                xi_mean = np.mean(Xi)
                xi_var = np.var(Xi)
                # p_list[k].append( [xi_mean, xi_var])
                p_label.append( [xi_mean, xi_var])
                 
            else  :
                # special handle for month , because 6,7 is missing
                if i == 1:
                    TOTAL_MON=12
                    Num_attris = TOTAL_MON
                else :
                    Num_attris = X_nonclassified.nunique()
                ret_list = []

                Xi_list = Xi.tolist()
                
                xi_min = np.min(X_nonclassified)
                # correct
                # print("xi_min", "i_", i, xi_min)
                STEP=1
                # [start, end, step]
                # for j in range(Num_attris):
                for j in range(xi_min, xi_min+Num_attris, STEP):
                    num_attri = Xi_list.count(j)
                    # print("X_nonclassified",X_nonclassified)
                    # print("i_",i, "j_",j, "count_", num_attri,"Num_attris", Num_attris )
                    p = (num_attri+1)/(total_num+Num_attris)
                    ret_list.append(p)

                # p_list[k].append(pd.Series(ret_list, index =list(range(xi_min, xi_min+Num_attris, STEP))))
                p_label.append(pd.Series(ret_list, index =list(range(xi_min, xi_min+Num_attris, STEP))))
                                    
        
        # k=k+1
        # print("***********p_label",p_label)
        p_list.append(p_label)
        # p0 , p
    return p_prior, p_list, type_list

def predict_eval_gaussian(p_prior, p_list, type_list, X, y):
    m, n = X.shape
    predict_list=[]
    acc=0
    for i in range(m):
        # 'numpy.float64' object cannot be interpreted as an integer
        Xi=X.iloc[i, :].astype(object)

        predict, _ = predict_single_gaussian(p_prior, p_list, type_list, Xi)
        if(predict==y[i]):
            acc+=1   
        predict_list.append(predict)
    acc_p = acc/len(X)         
    # print("Acc :",acc/len(X))
    return acc_p, predict_list
    # output label
def predict_single_gaussian(p_prior, p_list, type_list,  X):
    n = X.shape[0] #len(X)
    # X.shape 1, n 
    # 10|
    # print(n)
    x_p0 ,x_p1, x_p2=0,0,0
    
    x_list=[x_p0, x_p1,x_p2]
    NUM_P_LIST=3
    for i in range(NUM_P_LIST):
        # err
        x_list[i] =np.log(p_prior[i]) 
        for j in range(n):
            if(type_list[j]=="continuous"):
                mean, var = p_list[i][j]
                # np.log(0)
                x_list[i]+=np.log(1/np.sqrt(2*np.pi)*var * np.exp(-(X[j]-mean)**2/(2*var**2)))
            # mean0, var0 = p0_xi.conditional_pro
            # x_p1 += np.log(1 / (np.sqrt(2 * np.pi) * var1) * np.exp(- (x[i] - mean1) ** 2 / (2 * var1 ** 2)))
            # x_p0 += np.log(1 / (np.sqrt(2 * np.pi) * var0) * np.exp(- (x[i] - mean0) ** 2 / (2 * var0 ** 2)))
            # 3*10
            else:
                x_list[i]+=np.log(p_list[i][j][X[j]])
            # print( j,"_ ", X[j])
        x_list[i]=x_list[i].tolist()
    return LABEL_LIST[x_list.index(np.max(x_list))], x_list
# 


# # Train and test ncb

# In[18]:


import pandas as pd 
import numpy as np 
from sklearn.utils.multiclass import type_of_target
train_file_name=  '/home/aistudio/work/data/training sample.csv'
test_file_name=  '/home/aistudio/work/data/testing sample.csv'
another_test_file_name=  '/home/aistudio/work/data/another_testing sample.csv'
train_data= pd.read_csv(train_file_name)
test_data= pd.read_csv(test_file_name)
another_test_data= pd.read_csv(another_test_file_name)

train_X =train_data.iloc[:, :-1]
train_y =train_data.iloc[:, -1]
test_X =test_data.iloc[:, :-1]
test_y =test_data.iloc[:, -1]
a_test_X =another_test_data.iloc[:, :-1]
a_test_y =another_test_data.iloc[:, -1]
# print("X", X,y)

######  testing decrete algo#####

# print("#####For class low: ,", P_list[0])
# print("#####For class median: ,", P_list[1])
# print("#####For class high: ,", P_list[2])
# ret = predict_eval(P_prior, P_list,  test_X, test_y)
# print(ret)

######  testing gaussian#####

# ret = predict_eval_gaussian(P_prior, P_list, type_list, test_X, test_y)

# Part 1 
ACC_FORMAT="The accuracy on training data is __%r__. The accuracy on testing data is __%r___. "
P_prior, P_list, type_list= calc_condi_prob(train_X, train_y)
acc_train = predict_eval(P_prior, P_list, train_X, train_y)
acc_test = predict_eval(P_prior, P_list, test_X, test_y)
print("####Using discretization for continuous value: ")
print(ACC_FORMAT%(acc_train,acc_test))

P_prior_g, P_list_g, type_list_g= calc_condi_prob_gaussian(train_X, train_y)
print("**********P_list_g", P_list_g)
acc_train_g,_ = predict_eval_gaussian(P_prior_g, P_list_g, type_list_g, train_X, train_y)
acc_test_g,_ = predict_eval_gaussian(P_prior_g, P_list_g, type_list_g, test_X, test_y)
print("####Using gaussian for continuous value: ")
print(ACC_FORMAT%(acc_train_g,acc_test_g))
print("####testing another file")
print("another_file data: ", another_test_data)
acc_test_g, predict_list = predict_eval_gaussian(P_prior_g, P_list_g, type_list_g, a_test_X, a_test_y)
print("#####predict result is :",predict_list)




# In[81]:


#####Part2
CONDI_FORMAT=" P(low| sample %r)=__%r___. P(median |sample %r)=__%r__ ,P(high |sample %r)=__%r__" 
for i in range(1,6,1):
    test=test_X.iloc[i,:].astype(object)
    # predict_single_gaussian(p_prior, p_list, type_list,  X):
    _, ret = predict_single_gaussian(P_prior_g, P_list_g, type_list_g, test)
    # print(ret)
    print(CONDI_FORMAT%(i,np.exp( ret[0]),i,np.exp( ret[1]),i,np.exp( ret[2])))


# In[53]:


#Part 3
# output one label
# print("train_X.shape[0]",train_X.shape[0])
# LABEL_LIST
for i in range(3):
    for j in range(train_X.shape[1]):
        print("#####For  %rth attribute| Label %r "%(j,LABEL_LIST[i]))
        if type_list[j] =="continuous":
            print("###Using Gaussian: ",P_list_g[i][j])
            print("###Using discretization: ",P_list[i][j])
        else:
            print( P_list[i][j])


# In[ ]:




