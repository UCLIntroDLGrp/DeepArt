import os
import sys
sys.path.insert(0, os.path.realpath('../'))
import pickle
from Visualisation.trainingVisualisation import TrainingTracker

experiment2History = pickle.load( open( "../SavedData/Experiment2HistoryResnet.pickle", "rb" ) )




#Make performance plots
tracker = TrainingTracker()
tracker.addArray(experiment2History['acc'],'Training_Accuracy')
tracker.addArray(experiment2History['loss'],'Training_Cost')
tracker.addArray(experiment2History['val_acc'],'Validation_Accuracy')
tracker.addArray(experiment2History['val_loss'],'Validation_Cost')
tracker.makePlots("Experiment2HistoryPlotsResnet")