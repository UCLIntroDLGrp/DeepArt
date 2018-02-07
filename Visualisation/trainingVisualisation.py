import collections
import matplotlib.pyplot as plt

class TrainingTracker(object):
    '''
    Track training performance metrics
    '''
    def __init__(self):
        self.valDict = collections.defaultdict(lambda: [])

    def addArray(self, valArray, dataLabel = None):
        '''
        Add a full array of values in the tracker object under a metric label
        :param valArray: The array of values to add
        :param dataLabel: What are the values added
        :return: N/A
        '''

        if(dataLabel == None):
            raise ValueError("How do you want me to track data if you don't provide labels for them?")
        elif(not isinstance(dataLabel,str)):
            raise ValueError("Seriously, not a string?! ...")
        else:
            self.valDict[dataLabel] = valArray

    def makePlots(self, save=False):
        '''
        Plot the performance metrics.
        :return: N/A
        '''

        plt.figure(figsize=(20,20))
        number_of_plots = len(self.valDict.keys())

        i=1
        for k,v in self.valDict.items():
            plt.subplot(number_of_plots, 1, i)
            plt.plot(v)
            plt.xlabel("Epochs")
            plt.ylabel(k)
            plt.title(k +" vs Epoch Number")
            i+=1

        plt.tight_layout()
        if(save):
            plt.savefig("../SavedData/" + save)
        else:
            plt.show()



    def getValues(self, label):
        '''Get the values associated with the label'''
        return self.valDict[label]

    def getFullDict(self):
        '''Get the full tracker dict'''
        return self.valDict