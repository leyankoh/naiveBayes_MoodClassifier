class naiveBayes():
    """
    This is my classifier. To run it, the most important variables are:
    classSplitter() - this function returns a dictionary of classes and the data belonging to those classes
    classProbabilities() - this function returns a dictionary of classes and the % chance of the data falling into them
    get_conditional_probs() - using the splitClasses data, it calculates the conditional probability of every attribute
    get_predictions() - this is where we feed in the probabilities from the training data, and the real test data to get predictions

    For testing of accuracy, I have added the following functions:
    split_train_test() - this function splits the data into training and testing data, according to the percentage you want
    test_errors() - feed the split training/testing data in order to see accuracy of the classifier with given data.
    """

    def __init__(self, data, target):

        self.data = data
        self.target = target

    def split_train_test(self, data, target, percentage):
        """
        :param data: insert data rows (list)
        :param target: insert target rows (list)
        :param percentage: indicate how much of the data you want to use as a training set
        :return: four lists: data_train, target_train, data_test, target_test. Use as needed
        """

        if not len(data) == len(target):
            raise ValueError("Length of data must be the same as length of targets")

        get_length = int(len(data) * percentage)  # get the number of data points you would like/
        data_train = data[0:get_length]
        data_test = data[get_length:]
        target_train = target[0:get_length]
        target_test = target[get_length:]

        return data_train, target_train, data_test, target_test

    def classSplitter(self, data, target):
        """
        :param data: this should be the list of instances and their data
        :param target: target outcome
        :return: a dictionary of classes as the key and data belonging to the class as its values
        """
        classes = set(target)
        # indexes between data and target already match
        separatedClasses = {}
        for c in classes:
            indexValues = [i for i, j in enumerate(target) if j == c]  # get list of indexes where target = class
            datalist = []  # create a list to store all the data belonging in a single class
            for i in indexValues:
                datalist.append(data[i])
                separatedClasses[c] = datalist

        return separatedClasses

    def classProbabilities(self, target):
        """
        :param target: target is a list of target values from the data being fed in
        :return: a dictionary where the key = class and the value is the probability of that class occurring within the dataset
        """
        classes = set(target)
        n_instances = float(len(target))

        probabilities = {}
        for c in classes:
            num_Class = target.count(c)  # get the number of instances belonging to this class
            prob = float(num_Class / n_instances)  # get proportion

            probabilities[c] = prob

        return probabilities

    def get_conditional_probs(self, splitClasses):
        """
        :description: This function will calculate the conditional probability of every attribute.
        However, since attributes are boolean (can only be 0 or 1), this will JUST calculate the conditional probability of 0
        i.e. P(att1 = 0 | class = 0) and so on
        USE dictionary created by classSplitter() function as the argument here
        :param splitClasses: This is the dictionary created that contains the split of all classes
        i.e. format is {0: [[0, 1, 1, 0], [0, 0, 0, 1]] 1: ...} and so on
        :return: a dictionary of lists for conditional probabilities for every attribute row.
        for example, the key = class, value = [0.3, 0.2, 0.1] with value[0] reflecting
        P(attr1=0 | class = key) and so on
        """
        n_attribs = len(splitClasses[0][0]) # pick an arbitrary row to check the number of attribs in the dataset
        conditional_probs = {}
        for key in splitClasses.keys():
            num_Classes = float(len(splitClasses[key])) # get the total number of instances in the class
            prob_list = []
            for i in range(n_attribs): # for each attribute
                row = [row[i] for row in splitClasses[key] if row[i] == 0] # get list of instances of a single attribute that are 0
                n_rows = float(len(row))
                prob_0 = float(n_rows/num_Classes) #
                prob_list.append(prob_0) # returns a list that should be 26 attributes long with a conditional probability for each

            conditional_probs[key] = prob_list

        return conditional_probs

    def get_predictions(self, class_probs, conditional_probs, data):
        """
        :param class_probs: Use get_class_probabilities() to get the probabilities of falling into a single class
        :param conditional_probs: Use get_conditional_probs() function to get a list of probabilities for each class
        :param data: feed the data set. it should be just a single line (list)
        :param target: feed the target data here
        :return: given a set of data, what class it should belong to
        """
        # calculate p(attrib | class)
        n_attrib = len(data)  # get number of attributes

        max_P = {}  # a dictionary to store the values of P(data | class)
        for c in class_probs.keys():
            condiList = conditional_probs[c]  # returns a list of probabilities for each attribute where attribute = 0
            probabilities = []  # create a list to store all probabilities of each attribute
            for i in range(n_attrib):
                if data[i] == 0:
                    probabilities.append(condiList[i])
                else:
                    probabilities.append(1.0 - condiList[i])
            likelihood = reduce(lambda x, y: x * y, probabilities)  # total likelihood P(d|h)
            max_P[c] = likelihood * class_probs[c]

        predicted_Class = max(max_P.keys(), key=(lambda k: max_P[k]))

        return predicted_Class

    def test_errors(self, data_train, target_train, data_test, target_test):
        classSplit_test = self.classSplitter(data_train, target_train)  # returns a dictionary of split classes
        classProbs = self.classProbabilities(target_train)  # get dictionary of probabilities
        condiProbs = self.get_conditional_probs(classSplit_test)
        right = 0.0
        wrong = 0.0
        for i in range(len(data_test)):
            train_prediction = self.get_predictions(classProbs, condiProbs, data_test[i])

            if train_prediction == target_test[i]:
                right += 1.0
            else:
                wrong += 1.0

        accuracy = float(right/(right + wrong))

        return accuracy


