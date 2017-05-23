import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
#train%matplotlib inline

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def plotScatter(xvar, yvar,df):
    data = pd.concat([df[xvar],df[yvar]],axis=1)
    data.plot.scatter(x=xvar,y=yvar)#,ylim=(0,800000))
    plt.show()

def corrHeatMap(df):
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat,vmax=.8,square=True)
    plt.show()

def histomize(df,ftr):
    sns.distplot(df[ftr],fit=norm)
    fig = plt.figure()
    res = stats.probplot(df[ftr],plot=plt)
    plt.show()

def logify(df,ftr):
    df[ftr] = np.log(df[ftr]+1)
    
def boolilogify(df,ftr,newftr):
    df[newftr] = pd.Series(len(df[ftr]), index=df.index)
    df[newftr] = 0 
    df.loc[df[ftr]>0,newftr] = 1
    #transform data
    df.loc[df[newftr]==1,ftr] = np.log(df[ftr])

x_train = pd.read_csv("./data/train.csv")
x_test = pd.read_csv("./data/test.csv")

df_train = pd.concat((x_train.loc[:,'MSSubClass':'SaleCondition'],
                      x_test.loc[:,'MSSubClass':'SaleCondition']))


#remove missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

#feature engineering
df_train['TotalSF'] = df_train['1stFlrSF']+df_train['2ndFlrSF']

#correlation matrix
#corrHeatMap(df_train)
#logify(df_train,'SalePrice')
logify(df_train,'TotalSF')
logify(df_train,'GrLivArea')
logify(df_train,'1stFlrSF')
#df_train['Has2ndFlr']
#boolilogify(df_train,'TotalBsmtSF','HasBsmt')
#boolilogify(df_train,'2ndFlrSF','Has2ndFlr')

df_train = pd.get_dummies(df_train)

X_train = df_train[:x_train.shape[0]]
X_test = df_train[x_train.shape[0]:]
y = x_train.SalePrice

#y = df_train['SalePrice']
#df_train = df_train.drop('SalePrice',axis=1)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# Ridge
model_ridge = Ridge()
alphas = [0.005, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 100]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
print(cv_ridge.min())

#plot the thing
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

# Lasso
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print(rmse_cv(model_lasso).mean())

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

# predict the y's

lasso_preds = np.expm1(model_lasso.predict(X_test))
lasso_solution = pd.DataFrame({"id":test.Id, "SalePrice":lasso_preds})
lasso_solution.to_csv("lasso_sol.csv", index = False)





