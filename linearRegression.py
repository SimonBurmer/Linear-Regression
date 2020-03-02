import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import pickle
import  os

#implementing the linear regression algorithm to predict students final grade based on a series of attributes.
#This data set consists of 33 attributes for each student. https://archive.ics.uci.edu/

loc_dir = os.path.dirname(__file__)
#the separator default is "," but in this Dataset the separator is ";" thats why we have to determine sep=";
data = pd.read_csv(loc_dir +"/data/student-mat.csv", sep=";")

#to accurately use linear regression the data must correlate
#the columns "G1", "G2", "studytime", "failures", "absences" 
#seems to have a correlation to "G3", to proof this we plot the Data 
fig, axs = plt.subplots(2, 3)

axs[0, 0].scatter(data["failures"], data["G3"])
axs[0, 0].set_title('failures/G3')
axs[0, 0].set(xlabel='failures', ylabel='G3')
axs[0, 1].scatter(data["studytime"], data["G3"])
axs[0, 1].set_title('studytime/G3')
axs[0, 1].set(xlabel='studytime', ylabel='G3')
axs[0, 2].scatter(data["absences"], data["G3"])
axs[0, 2].set_title('absences/G3')
axs[0, 2].set(xlabel='absences', ylabel='G3')
axs[1, 0].scatter(data["G1"], data["G3"])
axs[1, 0].set_title('G1/G3')
axs[1, 0].set(xlabel='G1', ylabel='G3')
axs[1, 1].scatter(data["G2"], data["G3"])
axs[1, 1].set_title('G2/G3')
axs[1, 1].set(xlabel='G2', ylabel='G3')

plt.tight_layout()
plt.show()

#On the studytime plot you can't really see a relation between G3 and Studytime. 
#To make sure that there is a relation we calculate the probability that someone gets a good grade

print("-----------------------------------------------")
print("probability for a good grade:")
for i in range(1,5):
    testdata = data[["G3","studytime"]]
    longStudy= testdata.loc[(testdata["studytime"] == i)]
    longStudy_goodGrades= testdata.loc[(testdata["studytime"] == i) & (testdata["G3"] > 15)]
    result = len(longStudy_goodGrades)/len(longStudy)
    print("studytime: "+str(i)+ " probability: "+str(result))
print("-----------------------------------------------")


#select the important columns 
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

X = np.array(data.drop(["G3"], axis = 1)) #Features
Y = np.array(data["G3"]) #Labels 

#Train multiple models and save the best on
best_acc = 0
for _ in range(50): 
    #test_sitze = 0.1 --> 10% off the data is test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    model = linear_model.LinearRegression()

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)

    if acc > best_acc:
        best_acc = acc
        with open(loc_dir + "/Data/studentgrades.pickle", "wb") as f:
            pickle.dump(model, f)

#Load model 
pickle_in = open(loc_dir + "/Data/studentgrades.pickle", "rb")
model = pickle.load(pickle_in)


print("-----------------------------------------------")
print("Model facts:")
print("accuracy: " + str(model.score(x_test, y_test)))
print('Coefficient: \n', model.coef_) #steigung
print('Intercept: \n', model.intercept_)#Y-achsenabschnitt
print("----------------------------------------------")


print("----------------------------------------------")
print("show accuracy:")
print("    predicted   |    input data   |   real value")
predicted = model.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])
print("----------------------------------------------")