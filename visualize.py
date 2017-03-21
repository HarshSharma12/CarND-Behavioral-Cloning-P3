import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Read the data 
columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
data = pd.read_csv('data/driving_log.csv', names=columns)
#data['steering_angle'] = data['steering_angle'].convert_objects(convert_numeric=True)

print("Dataset Columns:", str(columns), "\n")
print("Shape of the dataset:", str(data.shape), "\n")
print(data.describe(), "\n")

print("Data loaded...")
#data = data[1:]
ds = np.array(data[1:].steering_angle).astype(np.float)

binwidth = 0.025

# histogram before image augmentation
plt.hist(ds,bins=np.arange(min(ds), max(ds) + binwidth, binwidth))
plt.title('Number of images per steering angle')
plt.xlabel('Steering Angle')
plt.ylabel('# Frames')
plt.show()
