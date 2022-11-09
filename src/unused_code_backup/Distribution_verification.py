import numpy as np
import math
import matplotlib.pyplot as plt


eig_val = np.load('eig_val.npy')
eig_vec = np.load('eig_vec.npy')
data_sample = np.load('data_sample.npy')
covered_each_axis = []
all_axis_data = []
plt.plot(eig_val)
plt.xlabel('num of pcs')
plt.ylabel('variance')
plt.show()
for j in range(512):
    axis_testing = j  # principal component axis we are testing
    data_along_eig_vec = []
    for i in range(len(data_sample+1)):
        # do a projection of all data point to the principal component axis we are testing
        data_along_eig_vec.append(np.dot(data_sample[i],eig_vec[axis_testing])/np.linalg.norm(eig_vec[axis_testing]))
    data_mean = np.asarray(data_along_eig_vec).mean()  # Calculate data mean for axis tested
    var = np.var(data_along_eig_vec) # Calculate data var for axis tested
    std = math.sqrt(var)
    count = 0
    coeff = 3
    for i in range(len(data_sample+1)):
        if (data_mean-coeff*std)<data_along_eig_vec[i]<(data_mean+coeff*std):
            count+=1
    covered = count/len(data_sample)
    covered_each_axis.append(covered)



print(covered_each_axis)
min = min(covered_each_axis)
max = max(covered_each_axis)
mean = np.mean(covered_each_axis)
print("max: {}, min: {}, mean: {}".format(max,min,mean))
#print(data_along_eig_vec)
plt.hist(covered_each_axis)
plt.show()