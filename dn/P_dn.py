import os,sys,glob
import cPickle 
import numpy as np
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.learning_rule import Momentum
from DBL_model import DBL_model
from DBL_util import paramSet
import scipy.io        
class Deep_dn(DBL_model):
    def __init__(self,algo_id,model_id,num_epoch,num_dim,train_id,test_id): 
       super(Deep_dn, self).__init__(algo_id,model_id,num_epoch,num_dim,train_id,test_id)
       self.setup()
        
    def setupParam(self):        
        # 17*17 denoising exp
        self.path_train = 'data/VOC_patch/'
        self.path_test = 'data/test/'
        self.dataset_id = 2            
        self.batch_size = 100
        #self.batch_size = 10000
        self.p_data = {'ds_id':2}   # dn data        

    def loadData_train(self):
        if self.train_id==0:
            self.p_data['mat_id']=[1]
            self.loadData(self.path_train,'train',range(int(1e6)))
            self.p_data['mat_id']=[2]
            self.loadData(self.path_train,'valid',range(int(1e5)))

        self.p_monitor['channel'] = ['train_objective','valid_objective']
        self.p_monitor['save'] = 'log'+self.dl_id

   
    def train(self):
        db = 0
        if not db:
            self.runTrain()
        else:
            # debugging
            #self.loadData(self.path_train,'valid',range(0,10,3))
            self.DataLoader.data['valid'] = self.DataLoader.data['train']
            self.test_valid()

    def test(self):
        if not os.path.exists('result/'+self.dl_id):
            os.mkdir('result/'+self.dl_id)
        if self.test_id==-1:
            self.loadData(self.path_train,'valid',int(1e5)+range(int(1e5)),options={'data_id':-1,'mat_id':[1]})
            self.DataLoader.data['test'] = self.DataLoader.data['valid']
            result = self.runTest(metric = 2)
            print result[1]
            print result[0][0].shape
        elif self.test_id==0:
            self.loadData(self.path_test,'test',options={'data_id':1,'data':'test_p10010.mat'})
            result = self.test(self.batch_size,1)
            scipy.io.savemat(self.result_mat,mdict={'result':result})
        elif self.test_id==1:
            self.p_data['data_id'] = 2
            self.p_data['data'] = 'berk_test.mat'
            self.test_multi(range(100))
        elif self.test_id==2:
            self.p_data['data_id'] = 2
            self.p_data['data'] = 'pop_test.mat'
            self.test_multi(range(7))

    def test_multi(self,ran):
        pre =self.result_mat[:-4]
        for i in ran:
            print "do: image "+str(i)
            self.p_data['im_id'] = i
            self.loadData(self.path_test,'test')
            result = self.runTest()
            scipy.io.savemat(pre+'_'+str(i)+'.mat',mdict={'result':result})

    def buildModel(self):
        self.ishape = Conv2DSpace(shape = (17,17),num_channels = 1)

        # 1. parameter        
        if self.model_id ==-1:
            # no hidden layer (linear)
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.01,layer_type=2)]]
        elif self.model_id ==0:
            # 1 tanh + 1 linear
            self.p_layers = [
                [self.param.param_model_fc(dim = self.num_dim[0],irange=0.01,layer_type=0),
                self.param.param_model_fc(dim = self.num_dim[1],irange=0.01,layer_type=2)]
                ]
        elif self.model_id ==1:
            # 1 tanh + 1 linear
            self.p_layers = [
                [self.param.param_model_fc(dim = self.num_dim[0],irange=0.01,layer_type=1),
                self.param.param_model_fc(dim = self.num_dim[1],irange=0.01,layer_type=2)]
                ]
        elif self.model_id ==2:
            # train 2 sigmoid layer            
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.01,layer_type=0),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.01,layer_type=0),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=0.01,layer_type=2)]]
        elif self.model_id ==3:        
            # train 2 tanh layer
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.01,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.01,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=0.1,layer_type=2)]]
        elif self.model_id ==4:
            # train 3 tanh layer
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.05,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.05,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=0.05,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[3],irange=0.05,layer_type=2)]]
        elif self.model_id ==5:
            # train 3 tanh layer
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.07,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.05,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=0.05,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[3],irange=0.05,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[4],irange=0.07,layer_type=2)]]

    def buildAlgo(self):
        print "epoch remaining:",self.num_epoch
        if self.algo_id == 0:
            algo_lr = 2*1e-5
            algo_mom = 1e-4
            if self.model_id == -2:
                algo_lr = 1e-3
                algo_mom = 1e-3
            elif self.model_id ==1:
                algo_lr = 2*1e-5
                algo_mom = 1e-3
            elif self.model_id ==3:
                algo_lr = 2*1e-5
                algo_mom = 1e-4
            elif self.model_id ==4:
                algo_lr = 2*1e-5
                algo_mom = 1e-4
            self.p_algo = self.param.param_algo(batch_size = self.batch_size,                    
                     monitoring_dataset = self.DataLoader.data,
                     termination_criterion=EpochCounter(max_epochs=self.num_epoch),
                     #cost=Dropout(input_include_probs={'l1': .8},input_scales={'l1': 1.}),
                     learning_rate = algo_lr,
                     learning_rule = Momentum(algo_mom),    
                     algo_type = self.algo_id)    
        else:
            algo_lsm = 'exhaustive'
            algo_cg = True # False
            algo_sstep = 1.0#1e-1,1e1
            algo_mia = 1e-4
            algo_ia = None #[1e-4]
            algo_rcg = True #False
            if self.model_id==-2:
                algo_lsm = 'exhaustive'
                algo_cg = True
            self.p_algo = self.param.param_algo(
                     batch_size = self.batch_size,                    
                     monitoring_dataset = self.DataLoader.data,
                     termination_criterion=EpochCounter(max_epochs=self.num_epoch),
                     conjugate = algo_cg,
                     scale_step = algo_sstep,
                     min_init_alpha = algo_mia,
                     init_alpha = algo_ia,
                     line_search_mode = algo_lsm,
                     reset_conjugate = algo_rcg,
                     algo_type = self.algo_id)    

    
"""
# test data/cost function
python P_MLP.py 0 -1 0 289 0
# test model 1
python P_MLP.py 0 1 1000 1000,289 0
# test model 3
python P_MLP.py 0 3 1000 500,500,289 0
"""


if __name__ == "__main__":             
    if len(sys.argv) != 7:
        raise('need six inputs: algo_id, model_id epoch_num layer train_id test_id')
    num_dim = sys.argv[4].split(',')
    num_dim = [int(x) for x in num_dim]
    exp = Deep_dn(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),num_dim,int(sys.argv[5]),int(sys.argv[6]))
    exp.run()
    """
    import cPickle
    nn = 'dl_p1_1_6.pkl' 
    a=cPickle.load(open(nn))
    """
    
