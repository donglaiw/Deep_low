import numpy as np
mm = 'ucb_0_15_3_1_1000_10_0'
#mm = 'ucb_0_15_2_1_5000_6_1'
a=np.load('../data/train/Xtrain_'+mm+'.npy')
b=np.load('../data/train/Ytrain_'+mm+'.npy')
print np.mean(np.power(b,2))
# cnn: 0.63 
print np.mean(np.power(b[:,0]-a[:,a.shape[1]/2],2))  
tmp = a[:,a.shape[1]/2];tmp[tmp<0.4]=0;
print np.mean(np.power(b[:,0]-tmp,2))  
# pb+cnn: 0.1577,0.1263,
