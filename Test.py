import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

um_epochs = 1
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    #0,1, 50K samples, 50% chance each chosen
    x = np.array([1 ,2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])  #np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    print(x)
    #shift 3 steps to the left
    y = np.roll(x, echo_step)
    print(y)
    #padd beginning 3 values with 0
    y[0:echo_step] = 0
    #Gives a new shape to an array without changing its data.
    #The reshaping takes the whole dataset and puts it into a matrix, 
    #that later will be sliced up into these mini-batches.
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))
    print(x)
    print(y)
    return (x, y)

data = generateData()

