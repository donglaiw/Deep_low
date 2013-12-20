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
    # 1. parameter
    DD = './voc/'
    ishape = Conv2DSpace(shape = (17,17),num_channels = 1)        
    param = SetParam()    
    
    p_sig = param.param_model_fc(dim = 289,irange=0.1,layer_type=0)    
    #p_cf = param.param_model_cf(n_classes = nclass,irange=0.1)        
    #p_linear = param.param_model_fc(dim = nclass,irange=0.1,layer_type=2)        
    p_fc = [p_sig]

    p_algo = param.param_algo(batch_size = 1000,
                             termination_criterion=EpochCounter(max_epochs=2),
                            #cost=Dropout(input_include_probs={'l1': .8},input_scales={'l1': 1.}),                             
                             learning_rate=0.001,
                             batches_per_iter =9,
                             init_momentum=0.5)
               

    net = CNN_NET(ishape)
    
    np.random.seed(1)
    rand_ind = np.random.permutation([i for i in range(100000)])
    net.loaddata(DD,2,nclass,rand_ind[:90000],rand_ind[90000:])
    net.setup([p_fc],p_algo)
    
    net.train()
