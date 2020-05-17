from django.shortcuts import render
from rest_framework import viewsets
from . models import approvals
from . serializers import approvalsSerializers
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status

###############################
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
import tensorflow as tf
import joblib
##############################


class ApprovalsView(viewsets.ModelViewSet):
    queryset = approvals.objects.all()
    serializer_class = approvalsSerializers



def credit(request):
    '''
    #############################################
    df = pd.read_csv('C:/Users/Yzat/Downloads/ML_Django/Credit_Approval/ML_Coding/bankloan.csv')
    df = df.dropna()
    df.isna().any()
    df = df.drop('Loan_ID', axis=1)
    df['LoanAmount']=(df['LoanAmount']*1000).astype(int)
    pre_y = df['Loan_Status']
    pre_X = df.drop('Loan_Status', axis=1)
    dm_X = pd.get_dummies(pre_X)
    dm_y = pre_y.map(dict(Y=1, N=0))
    smote = SMOTE('minority')
    X1, y = smote.fit_sample(dm_X, dm_y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(200, activation='relu', kernel_initializer='random_normal', input_dim=X_test.shape[1]))
    classifier.add(tf.keras.layers.Dense(400, activation='relu', kernel_initializer='random_normal'))
    classifier.add(tf.keras.layers.Dense(4, activation='relu', kernel_initializer='random_normal'))
    classifier.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, y_train, batch_size=20, epochs=50, verbose=0)
    eval_model = classifier.evaluate(X_train, y_train)
    classifier.save('yzat.h5')
    '''
    
    sc = MinMaxScaler()
    model = tf.keras.models.load_model('ml_app/credit.h5')
    X_random = pd.read_excel('ml_app/test.xlsx')
    X_random = sc.fit_transform(X_random)
    
    y_pred = model.predict(X_random)
    y_pred = (y_pred>0.50)
    
    a = y_pred
    ##############################################
    
    context = {
        'title': 'Anasayfa',
        'credit': 'Yzat',
        'list': a,
        'yzat': 'gdfgdfgdfgd'

    }
    return render(request, 'anasayfa.html', context)

@api_view(['GET', 'POST'])
def approvereject(request):
    try:
        
        
        # gives empty dict. why I do not know. ---------------------- 
        mydata=request.data
        unit=np.array(list(mydata.values()))
        print('begin************************')
        print(unit)
        print('end**************************')
        # --------------------------------------------------
        
        # generated list to see the code working
        mydata = [0,	3593,	4266,	132000,	180,	0,	0,	1,	0,	1,	1,	0,	1,	0,	1,	0,	0]
        mydata2 = [3,	3173,	0,	74000,	360,	1,	0,	1,	0,	1,	0,	1,	1,	0,	0,	1,	0]

        unit = np.array(mydata)
        unit=unit.reshape(1, -1)
        
        # loading saved ML model
        #mdl=tf.keras.models.load_model('C:/Users/Yzat/Downloads/ML_Django/Credit_Approval/ML_Coding/credit.h5')
        mdl=tf.keras.models.load_model('ml_app/credit.h5')
        
        # loading scalers to scale data
        #scalers=joblib.load("C:/Users/Yzat/Downloads/ML_Django/Credit_Approval/ML_Coding/scalers.pkl")
        scalers=joblib.load("ml_app/scalers.pkl")
        
        X=scalers.transform(unit)
        
        y_pred=mdl.predict(X)
        y_pred=(y_pred>0.58)
        newdf=pd.DataFrame(y_pred, columns=['Status'])
        newdf=newdf.replace({True:'Approved', False:'Rejected'})
        info = newdf.iloc[0][0]
        
        return JsonResponse('Your Status is {}'.format(info), safe=False)
    
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)