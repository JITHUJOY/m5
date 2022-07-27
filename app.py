# import Flask from flask module
from flask import Flask, request, render_template
import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import multiprocessing as mp
import gc
import datetime
import os
from sklearn.preprocessing import LabelEncoder
import calendar
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pickle
# import run_with_ngrok from flask_ngrok to run the app using ngrok

  
app =Flask(__name__, template_folder="templates/")
  
@app.route("/")
def hello():
     return render_template('home.html')

def get_week_number(x):
    """This Function is used to get weeknumber of particular date"""
    date=calendar.datetime.date.fromisoformat(x)
    return date.isocalendar()[1]


def get_season(x):
    """This function is used to get season in US according to various months"""
    if x in [12,1,2]:
        return 0      #"Winter"
    elif x in [3,4,5]:
        return 1   #"Spring"
    elif x in [6,7,8]:
        return 2   #"Summer"
    else:
        return 3   #"Autumn"


def month_start(x):
    """This is used to check if day is begining of month"""
    day=calendar.datetime.date.fromisoformat(x).day
    return 1 if day==1 else 0


def month_end(x):
    """This is used to check if day is end of month"""
    day=calendar.datetime.date.fromisoformat(x).day
    month=calendar.datetime.date.fromisoformat(x).month
    year=calendar.datetime.date.fromisoformat(x).year
    leap_yr=(year%4==0) # to check if it is a leap year
    val=(day==31 and month==1) or (day==29 if leap_yr else day==28) or (day==31 and month==3) or (day==30 and month==4) or\
        (day==31 and month==5) or (day==30 and month==6) or (day==31 and month==7) or (day==31 and month==8) or\
        (day==30 and month==9) or (day==31 and month==10) or (day==30 and month==11) or (day==31 and month==12)
    return 1 if val else 0

def year_start(x):
    """This is used to check if day is begining of year"""
    day=calendar.datetime.date.fromisoformat(x).day
    month=calendar.datetime.date.fromisoformat(x).month
    return 1 if (day==1 and month==1) else 0


def check_if_quater_begin(x):
    """This is used to check if day is begining of quater"""
    day=calendar.datetime.date.fromisoformat(x).day
    month=calendar.datetime.date.fromisoformat(x).month
    return 1 if (day==1 and (month in [1,4,7,9])) else 0


#Reference https://www.lawinsider.com/dictionary/quarter-end
def check_if_quater_end(x):
    """This is used to check if day is end of quater"""
    day=calendar.datetime.date.fromisoformat(x).day
    month=calendar.datetime.date.fromisoformat(x).month
    if (day==31 and month==3) or (day==30 and month==6) or (day==30 and month==9) or (day==31 and month==12):
        return 1
    else:
        return 0


def year_end(x):
    """This is used to check if day is end of year"""
    day=calendar.datetime.date.fromisoformat(x).day
    month=calendar.datetime.date.fromisoformat(x).month
    return 1 if (day==31 and month==12) else 0


@app.route('/predict', methods=['POST'])
def predict():
     item_id=request.form['item_id']
     calendar_data=pd.read_csv('data/calendar.csv')
     prices=pd.read_csv('data/sell_prices.csv')
     sales=pd.read_csv('data/sales_train_evaluation.csv')
     sales=sales[sales['id']==item_id]
     if(len(sales)>0):
       for i in range(1942,1970):
         sales['d_'+str(i)]=0
       l=[]
       for i in range(1,1970):
         l.append("d_"+str(i))
       df_sales=pd.melt(sales,id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],\
                 value_vars=l,var_name="d",value_name="sales")   
       df_final= pd.merge(df_sales, calendar_data, on='d', how='left')
       df_final= pd.merge(df_final, prices, on=['store_id','item_id','wm_yr_wk'], how='left') 
       lbl=LabelEncoder()
       df_final['item_id']=lbl.fit_transform(df_final['item_id'])   
       lbl=LabelEncoder()
       df_final['dept_id']=lbl.fit_transform(df_final['dept_id'])
       lbl=LabelEncoder()
       df_final['cat_id']=lbl.fit_transform(df_final['cat_id'])
       lbl=LabelEncoder()
       df_final['store_id']=lbl.fit_transform(df_final['store_id'])
       lbl=LabelEncoder()
       df_final['state_id']=lbl.fit_transform(df_final['state_id'])
       lbl=LabelEncoder()
       df_final['event_name_1']=lbl.fit_transform(df_final['event_name_1'])
       lbl=LabelEncoder()
       df_final['event_name_2']=lbl.fit_transform(df_final['event_name_2'])
       lbl=LabelEncoder()
       df_final['event_type_1']=lbl.fit_transform(df_final['event_type_1'])
       lbl=LabelEncoder()
       df_final['event_type_2']=lbl.fit_transform(df_final['event_type_2'])
       lbl=LabelEncoder()
       df_final['year']=lbl.fit_transform(df_final['year'])
       df_final.loc[df_final['state_id'] == 'CA', 'snap'] = df_final.loc[df_final['state_id'] == 'CA']['snap_CA']
       df_final.loc[df_final['state_id'] == 'TX', 'snap'] = df_final.loc[df_final['state_id'] == 'TX']['snap_TX']
       df_final.loc[df_final['state_id'] == 'WI', 'snap'] = df_final.loc[df_final['state_id'] == 'WI']['snap_WI']
       df_final.drop(['snap_CA','snap_TX','snap_WI'],axis=1,inplace=True)
       df_final.drop('weekday',axis=1,inplace=True)
       df_final.drop('wm_yr_wk',axis=1,inplace=True)
       df_final['week_number']=df_final['date'].apply(lambda x:get_week_number(x))
       df_final['season']=df_final['month'].apply(lambda x:get_season(x))
       df_final['quater_start']=df_final['date'].apply(lambda x:check_if_quater_begin(x))
       df_final['quater_end']=df_final['date'].apply(lambda x:check_if_quater_end(x))
       df_final['month_start']=df_final['date'].apply(lambda x:month_start(x))
       df_final['month_end']=df_final['date'].apply(lambda x:month_end(x))
       df_final['year_start']=df_final['date'].apply(lambda x:year_start(x))
       df_final['year_end']=df_final['date'].apply(lambda x:year_end(x))
       df_final['date']
       df=df_final
       df_final.fillna(0,inplace=True)
       df.fillna(0,inplace=True)
       gc.collect()
       df.sort_values(['id','date'],inplace=True)
       df=df.pivot_table(index=['item_id','store_id'],columns='date',values='sales')
       for aggregate in ['mean','std']:
         for shif in [28]:
            for r in [7,14,30,60,360]:
              roll=df.rolling(r,axis=1).agg(aggregate).shift(shif)
              dates=roll.columns
              name="roll_"+str(r)+"_shift_"+str(shif)+"_"+aggregate
              roll=roll.astype('float16')
              roll.reset_index(level=[0,1],inplace=True)
              roll=pd.melt(roll,id_vars=['item_id','store_id'],value_vars=dates,var_name='date',value_name=name)
              roll.fillna(-1,inplace=True)
              df_final=df_final.merge(roll,on=['item_id','store_id','date'])
              del roll
              gc.collect()
       roll=df.shift(28,axis=1).ewm(alpha=0.99,axis=1,adjust=False).mean()
       dates=roll.columns
       roll=roll.astype('float16')
       roll.reset_index(level=[0,1],inplace=True)
       roll=pd.melt(roll,id_vars=['item_id','store_id'],value_vars=dates,var_name='date',value_name='direct_ewm')
       roll.fillna(-1,inplace=True)
       df_final=df_final.merge(roll,on=['item_id','store_id','date'])
       for lag in range(28,100,7):
         i='direct_lag_'+str(lag)
         lag_i=df.shift(lag,axis=1)
         dates=lag_i.columns
         lag_i.reset_index(level=[0,1],inplace=True)
         lag_i=pd.melt(lag_i,id_vars=['item_id','store_id'],value_vars=dates,var_name='date',value_name=i)
         lag_i.fillna(-1,inplace=True)
         lag_i[i]=lag_i[i].astype('int16')
         df_final=df_final.merge(lag_i,on=['item_id','store_id','date'])
       del lag_i
       gc.collect()
       df_final['snap']=np.nan_to_num(df_final['snap'])
       df_final_pred=df_final.drop(["sales","id","d","date"],axis=1)
       filename = os.path.join('models','lgbm_whole_model_new.pkl')
       rf_model = joblib.load(filename)
       df_final_pred['pred_sales']=rf_model.predict(df_final_pred)  
       df_predictions=df_final_pred[-28:]['pred_sales']
       data={}
       ind=1
       for i in df_predictions:
         data['F'+str(ind)]=i
         ind=ind+1
       return render_template('output.html',data=data)     
     else:
       return render_template('error.html')
if __name__ == "__main__":
    app.run(debug=True)