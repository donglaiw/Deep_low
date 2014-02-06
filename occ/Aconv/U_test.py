import sys,os
import cPickle 
import numpy as np
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.learning_rule import Momentum
from DBL_model import DBL_model
from DBL_util import U_centerind

class Deep_occ_test(DBL_model):
    def __init__(self,algo_id,model_id,num_epoch,num_dim,train_id,test_id=-1): 
       super(Deep_occ_test, self).__init__(algo_id,model_id,num_epoch,num_dim,train_id,test_id)
       self.setup()
    
    def setupParam(self):        
        # 35*35*3 occ
        #self.ishape = Conv2DSpace(shape = (35,35),num_channels = 3)
        self.path_train = './'
        self.path_test = './'            
        self.p_data = {'ds_id':0}   # occ data         
        self.batch_size = 1000
        if self.train_id<=2:
            self.p_data['data']='db_'+str(self.train_id+1)+'.mat'
            self.p_data['data_id'] = 6
            print self.train_id
            self.ishape = Conv2DSpace(shape = (self.train_id+1,self.train_id+1),num_channels = 3)
            self.valid_set = [0]
            self.train_set = [0]

        if self.ishape.num_channels == 1:
            self.p_data['ishape']=self.ishape.shape
        else:
            self.p_data['ishape']= np.append(self.ishape.shape,self.ishape.num_channels)
            #print self.p_data['ishape']


    def loadData_train(self):        
        #train_id = range(1,31000,3)
        self.loadData(self.path_train,'train',self.train_set)
        self.loadData(self.path_train,'valid',self.valid_set)
        #print "ds:",self.DataLoader.data['train'].X.shape,self.DataLoader.data['train'].y.shape
        self.p_monitor['channel'] = ['train_objective','valid_objective','train_sm0_misclass','valid_sm0_misclass']
        self.p_monitor['save'] = 'log'+self.dl_id

    def train(self):
        db = 0
        if not db:
            self.runTrain()        
        else:
            # debugging
            #self.loadData(self.path_train,'valid',range(0,10,3))
            self.DataLoader.data['test'] = self.DataLoader.data['train']
            rr = self.runTest(metric=2)
            print rr[1]

    def test(self):
        import cPickle
        if self.test_id==0:
            nn = 'dl_p1_1_6.pkl' 
            a=cPickle.load(open(nn))
 
    def buildModel(self):
        num_in = np.prod(self.ishape.shape)*self.ishape.num_channels
        if self.model_id == 0:
            # 1 tanh + 1 softmax            
            ks = [[1,1],[2,2],[3,3]]
            ir = [0.5,0.05,0.05]
            ps = [[1,1],[1,1],[2,2]]
            pd = [[1,1],[1,1],[2,2]]
            kid = self.num_dim[0]
            n1 = 1
            crop_len = [(1+(self.ishape.shape[k]-ks[kid][k])/pd[kid][k])/ps[kid][k]  for k in [0,1] ]
            print ks[kid],self.ishape
            if max(crop_len)>1:
                crop_cen = [(self.ishape.shape[k]-crop_len[k])/2 for k in [0,1] ]
                #print crop_len,crop_cen
                self.p_data['crop_y'] = U_centerind(self.ishape.shape,crop_cen,crop_len)

            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[1],ks[kid],ps[kid],pd[kid],ir[kid],layer_type=0)],
                #[self.param.param_model_cf(n_classes = self.num_dim[2],irange=n1,layer_type=0)]
                [self.param.param_model_fc(dim = self.num_dim[2],irange=n1,layer_type=2)]
                ]

    def buildAlgo(self):
        if self.algo_id == 0:
            algo_lr = 1e-2
            algo_mom = 1e-1
            if self.model_id == -2:
                algo_lr = 1e-3
                algo_mom = 1e-3
            elif self.model_id ==1:
                algo_lr = 1e-4
                algo_mom = 1e-3
            elif self.model_id == 2:
                algo_lr = 1e-4
                algo_mom = 1e-3
            elif self.model_id ==3:
                algo_lr = 2*1e-5
                algo_mom = 1e-4
            elif self.model_id ==5:
                algo_lr = 1e-4
                algo_mom = 1e-3

            self.p_algo = self.param.param_algo(batch_size = self.batch_size,                    
                     termination_criterion=EpochCounter(max_epochs=self.num_epoch),
                     monitoring_dataset = self.DataLoader.data,
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
# linear layer: test data/cost function
python P_occ_conv.py 0 0 1 0 0 0

import cPickle;a=cPickle.load(open('dl_p0_1_0_0_3.pkl'))
"""


if __name__ == "__main__":             
    tid = int(sys.argv[1])
    if tid<=2:
        pkl_name = 'dl_p0_0_'+str(tid)+'_1_1_'+str(tid)+'_1.pkl'
        if os.path.exists(pkl_name):
            os.remove(pkl_name)
        exp = Deep_occ_test(0,0,1,[tid,1,1],tid,0)
        exp.run()
        from U_py import pkl2mat
        pkl2mat(pkl_name,'jo.mat')
    elif tid==3:
        pass
            
    
