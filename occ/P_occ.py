import sys,os
import cPickle 
import numpy as np
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.learning_rule import Momentum
from DBL_model import DBL_model

class Deep_occ(DBL_model):
    def __init__(self,algo_id,model_id,num_epoch,num_dim,test_id): 
       super(Deep_occ, self).__init__(algo_id,model_id,num_epoch,num_dim,test_id)
       self.setup()
    
    def setupParam(self):        
        # 35*35*3 occ
        #self.ishape = Conv2DSpace(shape = (35,35),num_channels = 3)
        self.path_train = 'data/train/'
        self.path_test = 'data/test/'            
        self.p_data = 0   # occ data         
        self.batch_size = 10000
    def loadData_train(self):        
        valid_id = range(0,31000,10)
        train_id = list(set(range(0,31000)).difference(set(valid_id)))
        self.loadData(self.path_train,'train',train_id)
        self.loadData(self.path_train,'valid',valid_id)
        self.p_monitor['channel'] = ['train_objective','train_sm0_misclass','valid_sm0_misclass']
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
        import scipy.io        
        self.loadWeight(self.param_pkl)
        if self.test_id==-1:
            self.loadData(self.path_test,'test',options={'data_id':1,'data':'test_im.mat'})
            self.test_sketchtoken(np.ones(self.DataLoader.data['test'].y.shape[0]))
        elif self.test_id==0:
            # 1 0 300 500,151: 0.67485
            # 1 1 300 300,200,151: 0.6957
            # 1 1 200 500,300,151: 0.7007
            # 1 1 300 500,300,151: 0.6957

            self.loadData(self.path_test,'test',options={'data_id':1,'data':'test_im.mat'})
            result = self.runTest(metric=0)
            self.test_sketchtoken(result[0][0])
            #scipy.io.savemat(self.result_mat,mdict={'result':result})
        elif self.test_id==1:
            if not os.path.exists('result/'+self.dl_id):
                os.mkdir('result/'+self.dl_id)
            pre =self.result_mat[:-4]
            for i in range(1,200):
                print "do: image "+str(i)
                self.loadData(self.path_test,'test',options={'data_id':2,'data':'dn_ucb.mat','im_id':i})
                result = self.runTest(metric=-1)
                scipy.io.savemat(pre+'_'+str(i)+'.mat',mdict={'result':result})
            #scipy.io.savemat(self.result_mat,mdict={'done':1})
    def buildModel(self):
        if self.model_id<=4:
            self.ishape = Conv2DSpace(shape = (1,3675),num_channels = 1)
        # 1. parameter        
        if self.model_id ==-1:
            # no hidden layer (linear)
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.01,layer_type=2)]]
        elif self.model_id ==-2:
            # softmax layer
            self.p_layers = [[self.param.param_model_cf(n_classes = self.num_dim[0],irange=0.01,layer_type=0)]]
        elif self.model_id ==0:
            # 1 tanh + 1 softmax
            self.p_layers = [
                [self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=1)],
                [self.param.param_model_cf(n_classes = self.num_dim[1],irange=0.01,layer_type=0)]
                ]
        elif self.model_id ==1:        
             # 1 tanh + 1 softmax
            self.p_layers = [
                [self.param.param_model_fc(dim = self.num_dim[0],irange=0.5,layer_type=1),
                 self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=1)],
                [self.param.param_model_cf(n_classes = self.num_dim[2],irange=0.01,layer_type=0)]
                ]
        elif self.model_id ==2:
            # train 2 sigmoid layer            
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=0),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=0)]]
        elif self.model_id ==3:        
            # train 2 tanh layer
            self.p_layers = [[self.param.param_model_fc(dim = self.num_dim[0],irange=0.1,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=0.1,layer_type=2)]]
    def buildAlgo(self):
        if self.algo_id == 0:
            algo_lr = 1e-4
            algo_mom = 1e-3
            if self.model_id == -2:
                algo_lr = 1e-3
                algo_mom = 1e-3
            elif self.model_id ==1:
                algo_lr = 1*1e-4
                algo_mom = 1e-3
            elif self.model_id ==3:
                algo_lr = 3*1e-5
                algo_mom = 1e-4
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

    def test_sketchtoken(self,yhat):
        yhat[yhat!=150] = 0
        yy = self.DataLoader.data['test'].y
        yy[yy!=150] = 0
        print float(sum(yhat==yy))/len(yy)
    
    def test_valid(self,metric=2):
        self.DataLoader.data['test'] = self.DataLoader.data['valid']
        result = self.test(self.batch_size,metric)        
        print result[1]
        print result[0][0].shape
"""
# linear layer: test data/cost function
python P_occ.py 0 -1 0 151 0
# softmax layer: test data/cost function
python P_occ.py 0 -2 10 151 0
# test model 1
python P_occ.py 0 1 1000 1000,289 0
# test model 3
python P_occ.py 0 3 1000 500,500,289 0
"""


if __name__ == "__main__":             
    if len(sys.argv) != 6:
        raise('need six inputs: algo_id, model_id epoch_num layer test_id')
    num_dim = sys.argv[4].split(',')
    num_dim = [int(x) for x in num_dim]
    exp = Deep_occ(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),num_dim,int(sys.argv[5]))
    exp.run()
    """
    import cPickle
    nn = 'dl_p1_1_6.pkl' 
    a=cPickle.load(open(nn))
    """
    
