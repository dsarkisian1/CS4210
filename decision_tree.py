#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from tkinter.tix import Y_REGION
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
for row in db:
  xRow = []
  j = 0
  for i in row:
    if j == 0:
      if row[0] == 'Young':
        xRow.append(1)
      elif row[0] == 'Prepresbyopic':
        xRow.append(2)
      else:
        xRow.append(3)
    elif j == 1:
      if row[1] == 'Myope':
        xRow.append(1)
      else:
        xRow.append(2)
    elif j == 2:
      if row[2] == 'Yes':
        xRow.append(1)
      else:
        xRow.append(2)
    elif j == 3:
      if row[3] == 'Reduced':
        xRow.append(1)
      else:
        xRow.append(2)
    elif j == 4:
      if row[4] == "No":
        Y.append(1)
      else:
        Y.append(2)
    j += 1
  X.append(xRow)
  


      


#transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy', )
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()