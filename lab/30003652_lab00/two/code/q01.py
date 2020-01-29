#python 3
# lab00 part 2 q01

import numpy as np

arr_2d=np.loadtxt('../data/shape_manipulation.txt', delimiter=',')

print('\nOriginal Array:\n',arr_2d)

def crop_array(arr_2d, offset_height, offset_width, target_height, target_width):
	return arr_2d[offset_height:offset_height + target_height, offset_width:offset_width + target_width]

cr_arr=crop_array(arr_2d,1,1,2,2)
print('\nCropped array:\n', cr_arr)

pad_arr=np.pad(cr_arr, 2, 'constant', constant_values=0.5)

print('\nPadded array:\n', pad_arr)

conc_arr=np.concatenate((pad_arr,pad_arr), axis=1)

print('\nConcatenated array: shape=', conc_arr.shape, '\n', conc_arr)

#end of code