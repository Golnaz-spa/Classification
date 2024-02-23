import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

df=pd.read_csv('blocks.csv')

print(df.head(3))
print(df.shape)
print(df.info())
#get summary of all columns. use include = "all" to get also summary of the categorical column
print(df.describe(include = "all"))
# finding missing values
print(df.isna().sum())

#number and proportions of classes in the response variable
print(df["block"].value_counts())
print(df["block"].value_counts()/df.shape[0])

#convert catagorical variable to numrical one
df['block'].replace(to_replace=['h_line', 'picture', 'text','v_line','graphic'], value=[1, 2, 3, 4, 5], inplace=True)

X = df.drop(['block'], axis= 1)
Y = df['block']

acclog=np.empty(1000)
accknn=np.empty(1000)
acclda=np.empty(1000)
accqda=np.empty(1000)
accgnb=np.empty(1000)
counter = 0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    #LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train,y_train)
    acclda[i] = lda.score(X_test, y_test)
    # QDA model
    qda = LinearDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    accqda[i] = qda.score(X_test, y_test)
    # Naïve Bayes model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    accgnb[i] = gnb.score(X_test, y_test)
    #Logistic Regression model
    lr=LogisticRegression(multi_class="multinomial",solver="newton-cg",max_iter=2000)
    lr.fit(X_train,y_train)
    acclog[i] = lr.score(X_test, y_test)
    #KNN model
    knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
    knn.fit(X_train, y_train)
    accknn[i] = lr.score(X_test, y_test)

print("maximum LDA is", np.amax(acclda))
print("maximum QDA is", np.amax(accqda))
print("maximum Naïve Bayes is", np.amax(accgnb))
print("maximum Logistic Regression is", np.amax(acclog))
print("maximum KNN is", np.amax(accknn))




