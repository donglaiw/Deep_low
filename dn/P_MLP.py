import os,sys
sys.path.append(os.path.expanduser('~/')+'ss/')
from U_const import *
sys.path.append(U_VLIB+'Donglai/DeepL/pylearn2')
sys.path.append(U_VLIB+'Donglai/DeepL/Deep_wrapper/pylearn2')
from DBL_test import *

if __name__ == "__main__": 
    # e.g. denoising 
    #ishape = VectorSpace(17,17,3)
    nclass = 17*17
    if U_LOCAL:
        DD = '/home/Stephen/Desktop/Deep_Low/dn/'
    else:
        DD = './voc/'
    param = param_dnn()
    
    ishape = Conv2DSpace(shape = (17,17),num_channels = 3)        
    """"""
    p_fc = param.param_model_fc(dim = 1000,irange=0.1)    
    p_cf = param.param_model_cf(n_classes = nclass,irange=0.1)        
    p_algo = param.param_algo(num_perbatch = 1000,num_epoch=100,rate_grad=0.001,rate_momentum=0.5)
                            

    net = CNN_NET(DD, ishape, [[p_fc],[p_cf]],p_algo)
    
    np.random.seed(1)
    rand_ind = np.random.permutation([i for i in range(100000)])
    net.loaddata(2,rand_ind[:90000],rand_ind[90000:])
    net.model()
    net.train()
