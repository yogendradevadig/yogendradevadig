# Import Data Set

import pandas as pd
data = pd.read_csv("NewspaperData.csv")
data.head()

data.info()

# Correlation

data.corr()

import seaborn as sns
sns.distplot(data['daily'])

import seaborn as sns
sns.distplot(data['sunday'])

Fitting a Linear Regression Model

import statsmodels.formula.api as smf
import seaborn as sns
model = smf.ols("sunday~daily",data = data).fit()

sns.regplot(x="daily", y="sunday", data=data);

#Coefficients
model.params

#t and p-Values
print(model.tvalues, '\n', model.pvalues)    

#R squared values
(model.rsquared,model.rsquared_adj)

# Predict for new data point

#Predict for 200 and 300 daily circulation
newdata=pd.Series([200,300])

data_pred=pd.DataFrame(newdata,columns=['daily'])

model.predict(data_pred)

