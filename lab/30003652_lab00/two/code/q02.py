#cs 763 lab00 part 2 q02 Code Vectorization

import numpy as np
from time import time
import cv2

arr=np.loadtxt('../data/code_vectorization.txt', delimiter=',')
arr=arr.astype(int)
print('\nOriginal array:\n', arr)

fil_arr=arr[(arr<=9) & (arr>=1)]

print('\nFiltered array:\n', fil_arr)

end_arr=np.zeros((fil_arr.size,fil_arr.max()+1))

end_arr[range(fil_arr.size),fil_arr]=1
end_arr=end_arr.astype(int)

print('\nOne hot encoded array:\n', end_arr)

n_2s=sum(end_arr[:,2])
print('\nNumber of 2s = ', n_2s)

#end of code