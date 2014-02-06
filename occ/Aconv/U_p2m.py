import sys
from U_py import pkl2mat
pkl_name = sys.argv[1]
mat_name = None
if len(sys.argv)>2:
    mat_name = sys.argv[2]

pkl2mat(pkl_name,mat_name)
