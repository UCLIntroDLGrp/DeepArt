import os
import sys
sys.path.insert(0, os.path.realpath('../'))
import pickle
from Visualisation.trainingVisualisation import TrainingTracker


# List of (file with history, name to save plot ) tuble entries
#HISTORY_METADATA = [ ("../SavedData/Experiment2HistoryResnet.pickle", "Experiment2HistoryPlotsResnet" ) ]
HISTORY_METADATA =[('../SavedData/Experiment2HistoryResnet2.pickle',"Experiment2HistoryPlotsResnet2.png"),('../SavedData/Experiment1CapsnetHistory.pickle',"Experiment1CapsnetPlots.png"),('../SavedData/Experiment1CustomNetHistory.pickle',"Experiment1HistoryPlotsCustomNet.png")]

for file_to_load , save_name in HISTORY_METADATA:
    print("Making the plot for {}".format(file_to_load))
    history = pickle.load( open(file_to_load , "rb" ) )


    #Make performance plots
    tracker = TrainingTracker()
    tracker.addArray(history['acc'],'Training_Accuracy')
    tracker.addArray(history['loss'],'Training_Cost')
    tracker.addArray(history['val_acc'],'Validation_Accuracy')
    tracker.addArray(history['val_loss'],'Validation_Cost')
    tracker.makePlots(save_name)