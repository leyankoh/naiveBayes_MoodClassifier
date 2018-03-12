from random import *
from naiveBayes import naiveBayes

# generate arrays of sample data
dailyMood = []

# generate 365 days worth of data
# this will take a format of [0, 1, 0, 1, 1, 0] - these are all the data variables
days = 365
while days > 0:
    randList = lambda n: [randint(0, 1) for b in range(1, n+1)] # code attained at http://code.activestate.com/recipes/577944-random-binary-list/
    oneDay = randList(20)
    dailyMood.append(oneDay)
    days -= 1

# let's assume there'll be 4 classes for our moods - angry, happy, sad, neutral
randMood = lambda n: [randint(0, 3) for b in range (1, n+1)]
target = randMood(365)

# test naive bayes on generated array
# instantiate object
clf = naiveBayes(dailyMood, target)

data_train, target_train, data_test, target_test = clf.split_train_test(dailyMood, target, 0.7) # use 70% of the data as training
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

# nasty. Only 22%. Would perform better with a real data-set though
