import os,sys,glob
import cPickle 
import numpy as np
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.learning_rule import Momentum
from DBL_model import DBL_model
from DBL_util import paramSet

class Deep_low(DBL_model):
    def __init__(self,exp_id,model_id,num_epoch,num_dim,test_id): 
        self.exp_id = exp_id
        self.model_id = model_id
        self.num_epoch = num_epoch
        self.num_dim = num_dim
        self.test_id = test_id
        self.setup()
        super(Deep_low, self).__init__(self.ishape,self.p_layers,self.dataset_id)
        self.runExp()
    
    def setup(self):        
        if self.exp_id == 0:
            # 35*35*3 occ
            #self.ishape = Conv2DSpace(shape = (35,35),num_channels = 3)
            self.ishape = Conv2DSpace(shape = (1,3675),num_channels = 1)
            self.path_train = 'data/train/'
            self.path_test = 'data/test/'            
            self.dataset_id = 0   # occ data         
            self.batch_size = 10000
        self.dl_id = str(self.exp_id)+'_'+str(self.model_id)+'_'+str(self.num_epoch)
        self.param_pkl = 'dl_p'+self.dl_id+'.pkl'
        self.result_mat = 'result/'+self.dl_id+'/dl_r'+str(self.test_id)+'.mat'
        self.param = paramSet()
        self.buildMLP()

    def runExp(self):
        if not os.path.exists(self.param_pkl):
            if self.exp_id == 0:
                np.random.seed(100)
                #rand_ind = np.random.permutation([i for i in range(100000)])       
                #rand_ind = [i for i in range(100000)]
                #self.loadData(self.path_train,'train',rand_ind[:900000])
                self.loadData(self.path_train,'train',range(0,30000,3))
                #self.loadData(self.path_train,'valid',rand_ind[900000:])
                #print self.DataLoader.data['train'].X.shape
                #print self.DataLoader.data['valid'].X.shape
                pre_dl_id = self.param_pkl[:self.param_pkl.rfind('_')+1]
                fns = glob.glob(pre_dl_id+'*.pkl')
                epoch_max = 0
                if len(fns)==0:
                    # first time to do it, load matlab prior
                    mat_init = 'init_p'+str(self.model_id)+'.mat'
                    if os.path.exists(mat_init):
                        print "load weight: ", mat_init 
                        self.loadWeight(mat_init)
                else:
                    for fn in fns:
                        epoch_id = int(fn[fn.rfind('_')+1:fn.find('.pkl')])
                        if (epoch_id>epoch_max and epoch_id<self.num_epoch):
                            epoch_max = epoch_id 
                    if epoch_max>0:
                        self.loadWeight(pre_dl_id+str(epoch_max)+'.pkl')

                p_monitor ={'channel':['train_objective','valid_objective'],'epoch':epoch_max,'save':'log'+self.dl_id}
                if self.model_id == -2:
                    algo_lr = 1e-4
                    algo_mom = 1e-3
                elif self.model_id ==1:
                    algo_lr = 1*1e-4
                    algo_mom = 1e-3
                elif self.model_id ==3:
                    algo_lr = 3*1e-5
                    algo_mom = 1e-4
                p_algo = self.param.param_algo(batch_size = self.batch_size,                    
                         termination_criterion=EpochCounter(max_epochs=self.num_epoch),
                         #cost=Dropout(input_include_probs={'l1': .8},input_scales={'l1': 1.}),
                         learning_rate = algo_lr,
                         learning_rule = Momentum(algo_mom))    
            
                db = 0
                if not db:
                    self.train(p_algo,p_monitor)        
                    #self.saveWeight(self.param_pkl)                
                else:
                    # debugging
                    #self.loadData(self.path_train,'valid',range(0,10,3))
                    self.DataLoader.data['valid'] = self.DataLoader.data['train']
                    self.test_valid()
        elif not os.path.exists(self.result_mat):
            import scipy.io        
            self.loadWeight(self.param_pkl)
            if self.exp_id==0:
                if self.test_id==0:
                    self.loadData(self.path_test,'test',options={'data_id':1,'data':'test_p10010.mat'})
                    result = self.test(self.batch_size,1)
                    scipy.io.savemat(self.result_mat,mdict={'result':result})
                elif self.test_id==1:
                    pre =self.result_mat[:-4]
                    for i in range(100):
                        print "do: image "+str(i)
                        self.loadData(self.path_test,'test',options={'data_id':2,'data':'berk_test.mat','im_id':i})
                        result = self.test(self.batch_size,1)
                        scipy.io.savemat(pre+str(i)+'.mat',mdict={'result':result})
                    scipy.io.savemat(self.result_mat,mdict={'done':1})

    def buildMLP(self):        
        # 1. parameter        
        if self.model_id ==-1:
            # no hidden layer
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=1,layer_type=2)]]
        elif self.model_id ==-2:
            # softmax layer
            self.p_layers = [[self.param.param_model_cf(n_classes = self.num_dim[0],irange=0.01,layer_type=0)]]
        elif self.model_id ==0:
            # train 1 sigmoid layer
            self.p_layers = [[
                self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=0),
                self.param.param_model_fc(dim = self.num_dim[1],irange=1,layer_type=2)
                ]]
        elif self.model_id ==1:        
           # train 1 tanh layer
             self.p_layers = [[
                self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=1),
                self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=2)
                ]]
        elif self.model_id ==2:
            # train 2 sigmoid layer            
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=0),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=0)]]
        elif self.model_id ==3:        
            # train 2 tanh layer
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=0.1,layer_type=2)]]
    def test_param(self):
        print 'test param:'
        param = self.model.layers[0].get_params()      
        aa = param[0].get_value()
        bb = param[1].get_value()
        print aa[:3,:3],bb[:10]
    
    def test_valid(self,metric=2):
        self.DataLoader.data['test'] = self.DataLoader.data['valid']
        result = self.test(self.batch_size,metric)        
        print result[1]
        print result[0][0].shape
"""
# linear layer: test data/cost function
python P_occ.py 0 -1 0 151 0
# softmax layer: test data/cost function
python P_occ.py 0 -2 100 151 0
# test model 1
python P_occ.py 0 1 1000 1000,289 0
# test model 3
python P_occ.py 0 3 1000 500,500,289 0
"""


if __name__ == "__main__":             
    if len(sys.argv) != 6:
        raise('need six inputs: exp_id, model_id epoch_num layer test_id')
    num_dim = sys.argv[4].split(',')
    num_dim = [int(x) for x in num_dim]
    exps = Deep_low(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),num_dim,int(sys.argv[5]))
    """
    import cPickle
    nn = 'dl_r0_1_10.pkl' 
    a=cPickle.load(open(nn))
    """
    
