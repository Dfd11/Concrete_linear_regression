import pandas as pd 
import numpy as np
import random as rnd
from matplotlib import pyplot as plt



params=[]
errores=[]

book=pd.read_excel('Concrete_Data_Yeh.xlsx')

excel=pd.DataFrame(book,columns=['Z_coarseaggregate','Z_fineaggregate','Z_slag','Z_cement','Z_csMPa'])

samples=excel.values # samples[instancia][parametro]

rnd.shuffle(samples)


def divide(lst,n):
    for i in range(0,len(lst),n):
        yield lst[i:i+n]

def evaluate(param,samp):
    #param : array of values that simbolizes the parameters of the model
    #samp : array of values to be evaluated along the parameters p0*x0 + p1*x1 + ...
    acum=0
    for i in range(len(param)):
        acum=acum+param[i]*samp[i]
    return acum

def MSE(param,x,y):
    #param : array of values that simbolizes the parameters of the model
    #x : inputs
    #y : real output
    acum=0
    for row in range(len(x)): #for each row in len(x) "number of samples"
        #print(evaluate(param,x[row])-y[row])
        acum=acum+(evaluate(param,x[row])-y[row])**2 # evaluate the x with the parameters to get a possible
                                                     # result then minus the true result squared and added
                                                     # to a summ 
    #print(acum)
    acum=acum/(2*len(x)) #divide the sum by the number of samples and by 2 which is optional
    return(acum) # returns the mean square error which can be graphed to see the progress

def gradient_descent(param,x,y,alpha):
    #param : array of values that simbolizes the parameters of the model
    #x : inputs
    #y : real output
    #alpha : learning rate
    #returns: new parameters corrected
    temp=param.copy() # creates a copy of the parameters to be modified
    for column in range(len(param)): # for each column which is said in the amount of parameters
        sum=0
        for row in range(len(x)): # for each sample
            #print(column,row,evaluate(param,x[row]),y[row],evaluate(param,x[row])-y[row],x[row][column])
            sum=sum+(evaluate(param,x[row])-y[row])*x[row][column]  #we evaluate to get a hyp and minus the real value
                                                                    # times the value of the input
        sum=sum*alpha/len(x) # we divide by the number of samples
        
        temp[column]=temp[column]-sum #and modify the copy with the error obtained
    
    return temp #we return the new set of parameters

def train(alpha,osamples,error):
    
    global errores
    samples=osamples.copy()
    #RANDOMIZES THE SAMPLES

    results=[]
    
    temp=[]
    _error_=[[],[],[],[],[],[],[],[],[]]
    #SEPARA LAS ENTRADAS Y LAS SALIDAS EN OSAMPLES
    for i in range(len(samples)):
        results.append(samples[i][len(samples[i])-1])
        temp.append(samples[i][0:len(samples[i])-1])
    samples=temp
    ###############################################
    

    #DIVIDES THE SAMPLES IN BLOCKS

    
    part=int(len(samples)/10)
    #print('part ',part)

    
    ##############################
    #CROSS VALIDATION
    for tset in range(9):
        epoch=0
        param=[]

        #INITIALIZE PARAMETERS (cada vez que se va a re-entrenar hay que inicializar los parametros)
        for i in range(len(samples[0])):
            #param.append(0.0)
            param.append(rnd.randint(0,100)/100) # must be changed to random number
        x=0
        y=0
        training=[]
        y_training=[]
        testing=[]
        y_testing=[]
        validation=[]
        y_validation=[]

        #SET SETS
        x = list(divide(samples,part)).copy() #divides the inputs in part parts
        y = list(divide(results,part)).copy() #divides the outpus in part parts
        testing=x[0] 
        #print('testing ',len(testing))   #assign the same test section through all cross validation
        y_testing=y[0]
        #print(x.index(x[0]))
        x.pop(0)
        y.pop(0)

        validation=x[tset] #changes by 1 the block used for validation
        y_validation=y[tset]
        x.pop(tset)
        y.pop(tset)

        training=np.concatenate(x[0:])
        #print('training ',len(training)) #the rest is testing
        y_training=np.concatenate(y[0:])

        _error_[tset].append(MSE(param,validation,y_validation)) #PRIMER ERROR
        print('TSET: ',tset,'EPOCH: ',epoch,'ERROR: ',_error_[tset][epoch]*100)
        param=gradient_descent(param,training,y_training,alpha)
        _error_[tset].append(MSE(param,validation,y_validation))
        epoch=epoch+1
        cambio=np.sqrt(((_error_[tset][epoch]-_error_[tset][epoch-1])/_error_[tset][epoch-1])**2)
        while((_error_[tset][epoch]>0.001)and(epoch<200)and((cambio>0.00001)or(error))):
            cambio=np.sqrt(((_error_[tset][epoch]-_error_[tset][epoch-1])/_error_[tset][epoch-1])**2)
            #cambio=((_error_[tset][epoch]-_error_[tset][epoch-1])/_error_[tset][epoch-1])
            print('TSET: ',tset,'EPOCH: ',epoch,'ERROR: ',_error_[tset][epoch]*100)
            param=gradient_descent(param,training,y_training,alpha)
            _error_[tset].append(MSE(param,validation,y_validation))
            epoch=epoch+1
        
        _error_[tset].append(MSE(param,testing,y_testing))
        print("TESTING: ",_error_[tset][len(_error_[tset])-1])

    errores.append(_error_) 
    return param

params.append(train(0.3,samples,0))
print(params)
for i in range(len(errores[0])):
    
    print('BLOCK: ',i,'TEST: ',errores[0][i][len(errores[0][i])-1]*100,'TRAIN: ',errores[0][i][len(errores[0][i])-2]*100)
for i in range(len(errores[0])):
    plt.plot(range(len(errores[0][i])),errores[0][i])

plt.show()