#Import Data
import pandas as pd
survey_data = pd.read_csv('D:\Google Drive\Training\Datasets\Call Center Data\Call_center_survey.csv')

#total number of customers
print(survey_data.shape)

#Column names
print(survey_data.columns)

#Print Sample data
pd.set_option('display.max_columns', None) #This option displays all the columns 

survey_data.head()

#Sample summary
summary=survey_data.describe()
round(summary,2)

#frequency counts table
survey_data['Overall_Satisfaction'].value_counts()
survey_data["Personal_loan_ind"].value_counts()
survey_data["Home_loan_ind"].value_counts()
survey_data["Prime_Customer_ind"].value_counts()


#Non numerical data need to be mapped to numerical data. 
survey_data['Overall_Satisfaction'] = survey_data['Overall_Satisfaction'].map( {'Dis Satisfied': 0, 'Satisfied': 1} ).astype(int)

#number of satisfied customers
survey_data['Overall_Satisfaction'].value_counts()

#Defining Features and lables, ignoring cust_num and target variable
features=list(survey_data.columns[1:6])
print(features)
#Preparing X and Y data
#X = survey_data[["Age", "Account_balance","Personal_loan_ind","Home_loan_ind","Prime_Customer_ind"]]
X=survey_data[features]
y = survey_data['Overall_Satisfaction']

#Building Tree Model
from sklearn import tree
DT_Model = tree.DecisionTreeClassifier(max_depth=2)
DT_Model.fit(X,y)

#Plotting the trees
#Unfortunately drawing a beautiful tree is not easy in python, Still
#you will need to install pydot
#use this command in your anaconda prompt: conda install -c anaconda pydot=1.0.28
#First run below command on Anaconda Conslole and Install pydotplus manulaly otherwise import pydotplus throws an error
#pip install pydotplus 

#Before drawing the graph below command on anaconda console
#pip install graphviz

from IPython.display import Image
from sklearn.externals.six import StringIO

import pydotplus
dot_data = StringIO()
tree.export_graphviz(DT_Model, #Mention the model here
                     out_file = dot_data,
                     filled=True, 
                     rounded=True,
                     impurity=False,
                     feature_names = features)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#Rules
print(dot_data.getvalue())

#LAB : Tree Validation
########################################
##########Tree Validation
#Tree Validation
predict1 = DT_Model.predict(X)
print(predict1)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y, predict1)
print(cm)

total = sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy = (cm[0,0]+cm[1,1])/total
print(accuracy)


#LAB: Overfitting
#LAB: The problem of overfitting
############################################################################ 
##The problem of overfitting
#Choosing Cp and Pruning

#Dataset: "Buyers Profiles/Train_data.csv"
#Task 1: Import both test and training data
import pandas as pd
#Dataset: "Buyers Profiles/Train_data.csv"
#Import both test and training data
train = pd.read_csv("D:\\Google Drive\\Training\\Datasets\\Buyers Profiles\\Train_data.csv")
test = pd.read_csv("D:\\Google Drive\\Training\\Datasets\\Buyers Profiles\\Test_data.csv")

##print train.info()
train.shape
test.shape

# Building the tree model.
# the data have string values we need to convert them into numerica values
train['Gender'] = train['Gender'].map( {'Male': 1, 'Female': 0} ).astype(int)
train['Bought'] = train['Bought'].map({'Yes':1, 'No':0}).astype(int)

test['Gender'] = test['Gender'].map( {'Male': 1, 'Female': 0} ).astype(int)
test['Bought'] = test['Bought'].map({'Yes':1, 'No':0}).astype(int)

##print train.info()
##print test.info()

from sklearn import tree

#Defining Features and lables
features = list(train.columns[:2])

X_train = train[features]
y_train = train['Bought']

#X_train

X_test = test[features]
y_test = test['Bought']

#training Tree Model
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)

#Plotting the trees
#Unfortunately drawing a beautiful tree is not easy in python, Still
#you will need to install pydot
#use this command in your anaconda prompt: conda install -c anaconda pydot=1.0.28
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file = dot_data,
                     feature_names = features,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

predict1 = clf.predict(X_train)
print(predict1)

predict2 = clf.predict(X_test)
print(predict2)

####Calculation of Accuracy and Confusion Matrix
#on the train data
from sklearn.metrics import confusion_matrix ###for using confusion matrix###
cm1 = confusion_matrix(y_train,predict1)
cm1
total1 = sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1 = (cm1[0,0]+cm1[1,1])/total1
accuracy1


#On Test Data
cm2 = confusion_matrix(y_test,predict2)
cm2
total2 = sum(sum(cm2))
#####from confusion matrix calculate accuracy
accuracy2 = (cm2[0,0]+cm2[1,1])/total2
accuracy2

#LAB: Pruning

#We will rebuild a new tree by using above data and see how it works by tweeking the parameteres we have..
dtree = tree.DecisionTreeClassifier(max_leaf_nodes = 10, 
                                    min_samples_leaf = 5, 
                                    max_depth= 5)
dtree.fit(X_train,y_train)

predict3 = dtree.predict(X_train)
predict4 = dtree.predict(X_test)

#Accuracy of the model that we created with modified model parameters.
#on the train data
from sklearn.metrics import confusion_matrix ###for using confusion matrix###
cm1 = confusion_matrix(y_train,predict3)
cm1
total1 = sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1 = (cm1[0,0]+cm1[1,1])/total1
accuracy1


#On Test Data
cm2 = confusion_matrix(y_test,predict4)
cm2
total2 = sum(sum(cm2))
#####from confusion matrix calculate accuracy
accuracy2 = (cm2[0,0]+cm2[1,1])/total2
accuracy2

