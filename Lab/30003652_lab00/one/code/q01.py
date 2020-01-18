#CS764: CV Lab submission
#ID: 30003652, Lab00
#python3

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('v', metavar='N', type=float, nargs='+')
parser.add_argument("--p", type=float)
args= parser.parse_args()
	
print(args.v,args.p)
def norm(v,p):
	if not p:
		p=2
	n=len(v)
	sq_v=power(v,p)
	sum_v=sum(sq_v)
	norm_v=sum_v**(1/p)
	return norm_v

def power(v_vector,power):
	return [ x**power for x in v_vector]

norm_v=norm(list(args.v),args.p)

print('Norm of', list(args.v), 'is', round(norm_v,2))

