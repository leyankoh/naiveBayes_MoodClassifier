import pandas as pd
from naiveBayes import naiveBayes

luis = pd.read_csv("haha.csv")

luis.dropna(axis=0, inplace=True)
luis.drop(['ACTIVITY'], axis=1, inplace=True)

data = luis.values.tolist()

target = [row[-1] for row in data]

for row in data:
    del row[-1]

clf = naiveBayes(data, target)

data_train, target_train, data_test, target_test = clf.split_train_test(data, target, 0.7) # use 70% of the data as training
# split class
# basically generates a dictionary where the key is a target class, and the values are the list of instances that fall in that class
splitClass = clf.classSplitter(data_train, target_train)

# get probabilities of each class appearing. This will probably be fumbled since we have a random set
classProbs = clf.classProbabilities(target_train)
condiProbs = clf.get_conditional_probs(splitClass)

# alright, now we have trained the set.
# let's test the effectiveness
accuracy = clf.test_errors(data_train, target_train, data_test, target_test)
print accuracy