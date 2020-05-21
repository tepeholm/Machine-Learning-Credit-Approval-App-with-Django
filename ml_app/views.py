import pickle
#from keras import backend as K
from collections import Counter, defaultdict

import joblib
#import numpy as np
###############################
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.backend import clear_session

from django.contrib import messages
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from rest_framework import status, viewsets
from rest_framework.response import Response

from .forms import ApprovalForm
from .models import approvals
from .serializers import approvalsSerializers

##############################


class ApprovalsView(viewsets.ModelViewSet):
	queryset = approvals.objects.all()
	serializer_class = approvalsSerializers

def ohevalue(df):
	ohe_col=joblib.load("ml_app/mlmodels/allcol.pkl")
	cat_columns=['Gender','Married','Education','Self_Employed','Property_Area']
	df_processed = pd.get_dummies(df, columns=cat_columns)
	newdict={}
	for i in ohe_col:
		if i in df_processed.columns:
			newdict[i]=df_processed[i].values
		else:
			newdict[i]=0
	newdf=pd.DataFrame(newdict)
	return newdf

def approvereject(unit):
	try:
	  
		mdl=tf.keras.models.load_model('ml_app/mlmodels/credit.h5')
		scalers=joblib.load("ml_app/mlmodels/scalers.pkl")
		X=scalers.transform(unit)
		y_pred=mdl.predict(X)
		y_pred=(y_pred>0.58)
		newdf=pd.DataFrame(y_pred, columns=['Status'])
		newdf=newdf.replace({True:'Approved', False:'Rejected'})

		clear_session()
		return (newdf.values[0][0],X[0])
	except ValueError as e:
		return (e.args[0])

def cxcontact(request):
	if request.method=='POST':
		form=ApprovalForm(request.POST)
		if form.is_valid():
				
				Firstname = form.cleaned_data['firstname']
				Lastname = form.cleaned_data['lastname']
				Dependents = form.cleaned_data['Dependents']
				ApplicantIncome = form.cleaned_data['ApplicantIncome']
				CoapplicantIncome = form.cleaned_data['CoapplicantIncome']
				LoanAmount = form.cleaned_data['LoanAmount']
				Loan_Amount_Term = form.cleaned_data['Loan_Amount_Term']
				Credit_History = form.cleaned_data['Credit_History']
				Gender = form.cleaned_data['Gender']
				Married = form.cleaned_data['Married']
				Education = form.cleaned_data['Education']
				Self_Employed = form.cleaned_data['Self_Employed']
				Property_Area = form.cleaned_data['Property_Area']
				p = approvals(firstname=Firstname, lastname=Lastname, dependants=Dependents, 
                  applicantincome=ApplicantIncome, coapplicatincome=CoapplicantIncome, loanamt=LoanAmount,
                  loanterm=Loan_Amount_Term,credithistory=Credit_History, gender=Gender, married=Married,
                  graduatededucation=Education, selfemployed=Self_Employed, area=Property_Area )
				p.save()
				myDict = (request.POST).dict()
				df=pd.DataFrame(myDict, index=[0])
				answer=approvereject(ohevalue(df))[0]
				Xscalers=approvereject(ohevalue(df))[1]
				print(answer)
				print(Xscalers)
				print('***************************', Firstname)
				messages.success(request,'{}'.format(answer))
	
	form=ApprovalForm()
				
	return render(request, 'cxform.html', {'form':form})

