import sys
fn = sys.argv[1]
id = int(sys.argv[2]) 
a= open(fn)
aa = a.readlines()
ll = [ float(aa[x][aa[x].find(': ')+1:]) for x in range(id,len(aa),4)]
import numpy as np
print min(ll),np.argmin(ll)

