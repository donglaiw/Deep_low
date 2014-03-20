import os,sys

param=sys.argv[1]
#python T_test.py "0 3 63 50,50,1,4 2"
for i in range(4):
    os.popen('THEANO_FLAGS=device=gpu'+str(i)+' python P_occ_conv.py '+param+' '+str(i+1)+' &')
