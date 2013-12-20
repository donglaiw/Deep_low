import os,sys
import cPickle 
import numpy as np
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter
from DBL_model import DBL_model
from DBL_util import SetParam

class Deep_low(DBL_model):
    def __init__(self,exp_id,model_id,num_epoch,num_dim): 
        self.exp_id = exp_id
        self.model_id = model_id
        self.num_epoch = num_epoch
        self.num_dim = num_dim
        self.setup()
        super(Deep_low, self).__init__(self.ishape,self.p_layers,self.dataset_id)
        self.runExp()
    
    def setup(self):        
        if self.exp_id == 0:
            # 17*17 denoising exp
            self.ishape = Conv2DSpace(shape = (17,17),num_channels = 1)
            self.path_train = 'data/VOC_patch/'
            self.path_test = 'data/test/'
            self.dataset_id = 2            
            self.batch_size = 1000
        
        self.param_pkl = 'dl_p'+str(self.exp_id)+'_'+str(self.model_id)+'_'+str(self.num_epoch)+'.pkl'
        self.result_mat = 'dl_r'+str(self.exp_id)+'_'+str(self.model_id)+'_'+str(self.num_epoch)+'.mat'
        self.param = SetParam()
        self.buildMLP()

    def runExp(self):
        if not os.path.exists(self.param_pkl):
            if self.exp_id == 0:
                np.random.seed(1)
                rand_ind = np.random.permutation([i for i in range(100000)])       
                self.loadData(self.path_train,'train',rand_ind[:90000])
                self.loadData(self.path_train,'valid',rand_ind[90000:])
                self.loadData(self.path_test,'test',options={'data_id':1,'data':'test_p10010.mat'})
                # l2 paramters
                import scipy.io
                mat = scipy.io.loadmat('l2opt.mat')
                
                p_algo = self.param.param_algo(batch_size = self.batch_size,
                                         termination_criterion=EpochCounter(max_epochs=self.num_epoch),
                                        #cost=Dropout(input_include_probs={'l1': .8},input_scales={'l1': 1.}),
                                         learning_rate=0.001,
                                         batches_per_iter =9,
                                         init_momentum=0.5)    
            self.train(p_algo)
            self.saveWeight(self.param_pkl)
        elif not os.path.exists(self.result_mat):
            import scipy.io        
            self.loadWeight(self.param_pkl)
            if self.exp_id==0:
                #self.loadData(self.path_test,'test',options={'data_id':1,'data':'test_p10010.mat'})
                #result = self.test(self.batch_size,1)
                for i in range(100):
                    print "do: image "+str(i)
                    self.loadData(self.path_test,'test',options={'data_id':2,'data':'berk_test.mat','im_id':i})
                    result = self.test(self.batch_size,1)
                    scipy.io.savemat(str(i)+self.result_mat,mdict={'result':result})

    def buildMLP(self):        
        # 1. parameter        
        if self.model_id ==-1:
            # no hidden layer
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=2)]]            
        elif self.model_id ==0:
            # train 1 sigmoid layer
            self.p_layers = [[
                self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=0),
                self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=2)
                ]]
        elif self.model_id ==1:        
           # train 1 tanh layer
             self.p_layers = [[
                self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=0),
                self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=2)
                ]]
        elif self.model_id ==2:        
            # train 1 maxout layer
            pass
        elif self.model_id ==3:
            # train 2 sigmoid layer            
            self.p_layers = [[self.param.param_model_fc(dim = num_kernels[0],irange=0.1,layer_type=0),
                    self.param.param_model_fc(dim = num_kernels[1],irange=0.1,layer_type=0)]]
        elif self.model_id ==4:        
            # train 2 tanh layer
            self.p_layers = [[self.param.param_model_fc(dim = num_kernels[0],irange=0.1,layer_type=1),
                    self.param.param_model_fc(dim = num_kernels[1],irange=0.1,layer_type=1)]]
        elif self.model_id ==5:        
            # train 2 maxout layer            
            pass


if __name__ == "__main__":             
    if len(sys.argv) != 5:
        raise('need three input: exp_id, model_id')
    num_dim = sys.argv[4].split(',')
    num_dim = [int(x) for x in num_dim]
    exps = Deep_low(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),num_dim)
    """
    import cPickle
    nn = 'dl_r0_1_10.pkl' 
    a=cPickle.load(open(nn))
    """
    
