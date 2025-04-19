import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# set seaborn as the default style for plots
import seaborn as sns; sns.set()
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from scipy.stats import beta
from scipy.stats import f


# load file(s) into Colab
uploaded = files.upload()

# read CSV file into a pandas dataframe
data = pd.read_csv('seeds4.csv', index_col=0)
print(data.head())

data.head()

# drop the target column
df = data.drop('class' , axis=1)
# standardize the features
df = (df - df.mean())/df.std()
# check the column names
df.columns
# overview of the dataset columns and types
df.info()

Y=data["class"]

# get observation indices and variable names
observations = list(df.index)
variables = list(df.columns)

# box plot for each feature
ax = sns.boxplot(data=df, orient="v", palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

# pairwise scatter plots of all features
plt.figure()
sns.pairplot(df)
plt.title('Pairplot')

# plot the covariance matrix
dfc = df - df.mean()
plt. figure()
ax = sns.heatmap(dfc.cov(), center=0,cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=True, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title('Covariance matrix')

# run PCA
pca = PCA()
pca.fit(df)
Z = pca.fit_transform(df)

# scatter plot of the first two PCs
plt. figure()
plt.scatter(Z[:,0], Z[:,1], c='r')
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')

# plot PCA component vectors
A = pca.components_.T 
plt. figure()
plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('$A_1$')
plt.ylabel('$A_2$');
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom')

# 2D plot with color and size encoded by PC3 and PC4
plt. figure()
plt.scatter(A[:, 0],A[:, 1],marker='o',c=A[:, 2],s=A[:, 3]*500,
    cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(variables,A[:, 0],A[:, 1]):
    plt.annotate(label,xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

# get eigenvalues
Lambda = pca.explained_variance_

# plot scree plot
plt. figure()
x = np.arange(len(Lambda)) + 1
plt.plot(x,Lambda/Lambda.sum(), 'ro-', lw=2)
plt.xticks(x, [""+str(i) for i in x], rotation=0)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

(Lambda[1]+Lambda[0])/Lambda.sum()

# bar chart of explained variance ratios
ell = pca.explained_variance_ratio_
plt. figure()
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

# generate biplot with PC1 and PC2
A1 = A[:,0] 
A2 = A[:,1]
A3 = A[:,2]
Z1 = Z[:,0] 
Z2 = Z[:,1]
Z3 = Z[:,2]

fig, ax = plt.subplots()

for i in range(len(A1)):
    ax.arrow(0, 0, A1[i]*max(Z1), A2[i]*max(Z2),
              color='black', width=0.00005, head_width=0.0025)
    ax.text(A1[i]*max(Z1)*1.2, A2[i]*max(Z2)*1.2, variables[i], color='black')

for i in data['class'].unique():
    ax.scatter(Z1[data['class']==i], Z2[data['class']==i], marker='o',label=str(i))

legend = ax.legend(shadow=False, ncol=3, bbox_to_anchor=(0.85, -0.1))

plt.show()

# heatmap showing how features contribute to PCs
comps = pd.DataFrame(A,columns = variables)
sns.heatmap(comps,cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=True, square=True)
ax.tick_params(labelbottom=False,labeltop=True)
plt.title('Principal components')

# Hotelling’s T2 for multivariate control
alpha = 0.05
p=Z.shape[1]
n=Z.shape[0]

UCL=((n-1)**2/n )*beta.ppf(1-alpha, p / 2 , (n-p-1)/ 2)
UCL2=p*(n+1)*(n-1)/(n*(n-p) )*f.ppf(1-alpha, p , n-p)
Tsquare=np.array([0]*Z.shape[0])
for i in range(Z.shape[0]):
  Tsquare[i] = np.matmul(np.matmul(np.transpose(Z[i]),np.diag(1/Lambda) ) , Z[i])

fig, ax = plt.subplots()
ax.plot(Tsquare,'-b', marker='o', mec='y',mfc='r' )
ax.plot([UCL for i in range(len(Z1))], "--g", label="UCL")
plt.ylabel('Hotelling $T^2$')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))

fig.show()

# display data points flagged as out of control
print (np.argwhere(Tsquare>UCL))

# control chart for first principal component
fig, ax = plt.subplots()
ax.plot(Z1,'-b', marker='o', mec='y',mfc='r' , label="Z1")
ax.plot([3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label="UCL")
ax.plot([-3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label='LCL')
ax.plot([0 for i in range(len(Z1))], "-", color='black',label='CL')
plt.ylabel('$Z_1$')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))

fig.show()

# classification with logistic regression and naive bayes

logisticRegr = LogisticRegression(solver='lbfgs')
scoring=['accuracy']
scores_lr_full_data = cross_validate(logisticRegr, df, Y,cv=5, scoring=scoring)
scores_lr_Z = cross_validate(logisticRegr, Z, Y,cv=5, scoring=scoring)
scores_lr_Z12 = cross_validate(logisticRegr, Z[:,:2], Y,cv=5, scoring=scoring)

gnb = GaussianNB()
scores_gnb_full_data = cross_validate(gnb, df, Y,cv=5, scoring=scoring)
scores_gnb_Z = cross_validate(gnb, Z, Y,cv=5, scoring=scoring)
scores_gnb_Z12 = cross_validate(gnb, Z[:,:2], Y,cv=5, scoring=scoring)

scores_dict={}
for i in ['fit_time','test_accuracy']:
  scores_dict["lr_full_data " + i ]=scores_lr_full_data[i]
  scores_dict["lr_Z  " + i ]=scores_lr_Z[i]
  scores_dict["lr_Z12 " + i ]=scores_lr_Z12[i]
  scores_dict["gnb_full_data " + i ]=scores_gnb_full_data[i]
  scores_dict["gnb_Z " + i ]=scores_gnb_Z[i]
  scores_dict["gnb_Z12 " + i ]=scores_gnb_Z12[i]

scores_data=pd.DataFrame(scores_dict)
print(scores_data)

# show coefficients for logistic regression on original and PCA-transformed data
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
score = logisticRegr.score(X_test, y_test)
coefficient_full = logisticRegr.coef_

Z_train, Z_test, yz_train, yz_test = train_test_split(Z, Y, test_size=0.2)
logisticRegr_z = LogisticRegression()
logisticRegr_z.fit(Z_train, yz_train)
score_z = logisticRegr_z.score(Z_test, yz_test)
print(score_z)
coefficient_PCA = logisticRegr_z.coef_
np.around(coefficient_full, decimals=2)

np.around(coefficient_PCA, decimals=2)
