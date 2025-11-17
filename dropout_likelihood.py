import joblib

import pandas as pd

model=joblib.load('model.pkl')

scalar=joblib.load('scalar1.pkl')

encoder=joblib.load('encoder1.pkl')
income=14000
cgpa=6.64
motivation=4
pv_score=6
sibiling_count=2
parents_employeed=0
school=0
gender='Female'


df=pd.DataFrame([{'Family_Income_Monthly':income,'Current_CGPA':cgpa,'Student_Motivation':motivation,'Physical_Verification_Score':pv_score,'Siblings_Count':sibiling_count,'Parents_Employed':parents_employeed,'School_Type_Public':school,'Gender':gender}])

numeric=['Family_Income_Monthly','Current_CGPA','Student_Motivation','Physical_Verification_Score','Siblings_Count','Parents_Employed']

categorial=['Gender']

df[numeric]=scalar.transform(df[numeric])

df[categorial]=encoder.transform(df[categorial])


pred=model.predict(df)

pred_pro=model.predict_proba(df)[:,1][0]

dropout_likelihood=int(pred_pro*100)
print('dropout_likelihood',dropout_likelihood,'%')

retension_score=100-dropout_likelihood

print('retension score',retension_score,'%')








