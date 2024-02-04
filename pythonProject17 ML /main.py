import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('/Users/aidaabilzhanova/documents/clustered_data.csv')


target_variable = '2019 [YR2019]'


if data[target_variable].isnull().any():
    print(f"Warning: Missing values found in the target variable '{target_variable}'.")



predictors = data[['1995 [YR1995]', '1996 [YR1996]', '1997 [YR1997]', '1998 [YR1998]',
                  '1999 [YR1999]', '2000 [YR2000]', '2001 [YR2001]', '2002 [YR2002]',
                  '2003 [YR2003]', '2004 [YR2004]', '2005 [YR2005]', '2006 [YR2006]',
                  '2007 [YR2007]', '2008 [YR2008]', '2009 [YR2009]', '2010 [YR2010]',
                  '2011 [YR2011]', '2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]',
                  '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']]


X_train, X_test, y_train, y_test = train_test_split(predictors, data[target_variable], test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")



new_data = pd.DataFrame({'1995 [YR1995]': [10.0],
                         '1996 [YR1996]': [15.0],
                         '1997 [YR1997]': [20.0],
                         '1998 [YR1998]': [25.0],
                         '1999 [YR1999]': [30.0],
                         '2000 [YR2000]': [35.0],
                         '2001 [YR2001]': [40.0],
                         '2002 [YR2002]': [45.0],
                         '2003 [YR2003]': [50.0],
                         '2004 [YR2004]': [55.0],
                         '2005 [YR2005]': [60.0],
                         '2006 [YR2006]': [65.0],
                         '2007 [YR2007]': [70.0],
                         '2008 [YR2008]': [75.0],
                         '2009 [YR2009]': [80.0],
                         '2010 [YR2010]': [85.0],
                         '2011 [YR2011]': [90.0],
                         '2012 [YR2012]': [95.0],
                         '2013 [YR2013]': [100.0],
                         '2014 [YR2014]': [105.0],
                         '2015 [YR2015]': [110.0],
                         '2016 [YR2016]': [115.0],
                         '2017 [YR2017]': [120.0],
                         '2018 [YR2018]': [125.0],
                         '2019 [YR2019]': [130.0]})

predicted_value = model.predict(new_data)
print(f"Predicted Value: {predicted_value[0]}")
