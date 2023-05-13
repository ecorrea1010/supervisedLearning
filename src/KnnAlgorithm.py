import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

global data
def getData():
    global data
    data = pd.read_csv('files/covid.csv')

def prepareData():
    global data
    #print(data.isna().sum())
    features = [
        'Dry-Cough',
        'Difficulty-in-Breathing',
        'Sore-Throat',
        'Runny-Nose',
        'Fever',
        'Tiredness'
    ]
    target = ['Infected']
    independentTraining, independentTest, dependentTraining, dependentTest = train_test_split(data[features], data[target], test_size=0.3, random_state=0)
    return {
        'independentTraining': independentTraining,
        'independentTest': independentTest,
        'dependentTraining': dependentTraining,
        'dependentTest': dependentTest
    }

def model(dataModel):
    knn = KNeighborsClassifier(n_neighbors=7)
    dependentTraining = np.ravel(dataModel['dependentTraining'])
    knn.fit(dataModel['independentTraining'], dependentTraining)
    prediction = knn.predict(dataModel['independentTest'])
    accuracy = knn.score(dataModel['independentTest'], dataModel['dependentTest'])
    return {
        'prediction': prediction,
        'accuracy': accuracy
    }

def message(prediction):
    numberOfPredictions = [int(predictions) for predictions in prediction['prediction']]
    accuracyRound = round(prediction['accuracy'], 2)
    effectiveness = accuracyRound * 100
    infected = numberOfPredictions.count(1)
    noInfected = numberOfPredictions.count(0)
    message = 'This is the prediction of the model:'
    message += '\n'
    message += 'Infected: ' + str(infected)
    message += '\n'
    message += 'No infected: ' + str(noInfected)
    message += '\n'
    message += 'The effectiveness of the model was: ' + str(effectiveness) + '%'
    print(message)

def run():
    getData()
    prediction = model(prepareData())
    message(prediction)

if __name__ == '__main__':
    run()