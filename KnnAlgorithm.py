import numpy as np
import pandas as pd

class KnnAlgorithm:
    def __init__(self):
        self.features = None
        self.target = None
        self.cost = None

    def fit(self, features, target):
        self.features = features # x_train
        self.target = target.to_numpy().reshape(-1,1) # y_train
        # self.cost = np.zeros(len(self.target)).reshape(-1,1) #rastojanje

    def predict(self, features, num_of_neighbors):
        if num_of_neighbors%2 == 0: num_of_neighbors = num_of_neighbors+1
        self.cost = np.zeros((len(self.features), len(features)))
        ret_val = []
        # print(self.features.flour)
        for i in range(0, len(features)):
            for j in range (0, len(self.features)):
                # self.cost  = np.zeros((len(self.features), len(features)))
                self.cost[j][i] = np.sqrt(np.power(self.features.iloc[j,0]-features.iloc[i,0], 2)
                                          + np.power(self.features.iloc[j,1]-features.iloc[i,1], 2) 
                                          + np.power(self.features.iloc[j,2]-features.iloc[i,2], 2) 
                                          + np.power(self.features.iloc[j,3]-features.iloc[i,3], 2) 
                                          + np.power(self.features.iloc[j,4]-features.iloc[i,4], 2) 
                                          + np.power(self.features.iloc[j,5]-features.iloc[i,5], 2)
                                          )

        for i in range (0, len(features)):
            x = np.argsort(self.cost[:,i])[:num_of_neighbors]
            cnt_muffin = 0
            cnt_cupcake = 0
            for j in x:
                # print(self.target[j])
                if (self.target[j] == 'muffin'): cnt_muffin = cnt_muffin + 1
                if (self.target[j] == 'cupcake'): cnt_cupcake = cnt_cupcake + 1

            if (cnt_cupcake>cnt_muffin):
                ret_val.append('cupcake')
            else: 
                ret_val.append('muffin')               
        
        return ret_val