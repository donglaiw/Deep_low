import sys,os
import cPickle 
import numpy as np
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.learning_rule import Momentum
from DBL_model import DBL_model
from DBL_util import U_centerind

class Deep_occ(DBL_model):
    def __init__(self,algo_id,model_id,num_epoch,num_dim,train_id,test_id=-1): 
       super(Deep_occ, self).__init__(algo_id,model_id,num_epoch,num_dim,train_id,test_id)
       self.setup()
    
    def setupParam(self):        
        # 35*35*3 occ
        #self.ishape = Conv2DSpace(shape = (35,35),num_channels = 3)
        self.path_train = 'data/train/'
        self.path_test = 'data/test/'            
        self.p_data = {'ds_id':0}   # occ data         
        self.batch_size = 100
        if self.train_id<=5:
            self.p_data['data']='train_im.mat'
            self.p_data['data_id']=self.test_id
            # 2: 151-way classification
            # 3: 2-way classification
            if self.train_id<=4:
                self.ishape = Conv2DSpace(shape = (1,3675),num_channels = 1)
                # regression
            else:
                # conv
                self.ishape = Conv2DSpace(shape = (35,35),num_channels = 3)
            if self.train_id ==4:
                #crop output
                self.p_data['im_id'] = 1   # occ data         
                pshape = (35,35) 
                crop_len = (17,17)
                crop_cen = [(pshape[k]-crop_len[k])/2 for k in [0,1] ]
                self.p_data['crop_y'] = U_centerind(pshape,crop_cen,crop_len)

        elif self.train_id<=7:
            self.p_data['data']='train_feat_s.mat'
            self.p_data['data_id']=0
            self.ishape = Conv2DSpace(shape = (1,3000),num_channels = 1)
        elif self.train_id<=12:
            # gray 
            self.p_data['data']='train_img.mat'            
            self.ishape = Conv2DSpace(shape = (1,1225),num_channels = 1)
            if self.train_id<=10:
                # classification
                self.p_data['data_id'] = 9-self.train_id
            else:
                # gray regression
                self.p_data['data_id'] = self.train_id-8
                if self.train_id==12:
                    #crop output
                    pshape = (35,35) 
                    crop_len = (17,17)
                    crop_cen = [(pshape[k]-crop_len[k])/2 for k in [0,1] ]
                    self.p_data['crop_y'] = U_centerind(pshape,crop_cen,crop_len)
        if self.ishape.num_channels != 1:
            self.p_data['ishape']=self.ishape.shape
        else:
            self.p_data['ishape']= np.append(self.ishape.shape,self.ishape.num_channels)


    def loadData_train(self):        
        """
        """
        valid_id = range(0,31000,10)
        train_id = list(set(range(0,31000)).difference(set(valid_id)))
        #train_id = range(1,31000,3)
        self.loadData(self.path_train,'train',train_id)
        self.loadData(self.path_train,'valid',valid_id)
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
        import scipy.io        
        self.loadWeight(self.param_pkl)
        if self.test_id==-2:
            self.p_data['data'] = 'db.mat'
            self.p_data['data_id'] = 9-self.train_id
            self.loadData(self.path_test,'test')
            result = self.runTest(metric=-1)
            print result[0]
            #scipy.io.savemat('db.mat',mdict={'result':result})
           #single mat
        elif self.test_id==-1:
            # baseline
            self.test_sketchtoken(np.ones(self.DataLoader.data['train'].y.shape[0]))
            self.test_sketchtoken(np.ones(self.DataLoader.data['valid'].y.shape[0]))
        elif self.test_id==0:
            # 0,1 classification
            # 1 0 300 500,151: 0.67485
            # 1 1 300 300,200,151: 0.6957
            # 1 1 200 500,300,151: 0.7007
            # 1 1 300 500,300,151: 0.7097
            valid_id = range(0,31000,10)
            train_id = list(set(range(0,31000)).difference(set(valid_id)))
            #train_id = range(1,31000,3)
            self.loadData(self.path_train,'train',train_id)
            self.loadData(self.path_train,'valid',valid_id)
            result = self.runTest(self.DataLoader.data['train'])
            self.test_sketchtoken(result[0][0],self.DataLoader.data['train'].y)
            result = self.runTest(self.DataLoader.data['valid'])
            self.test_sketchtoken(result[0][0],self.DataLoader.data['valid'].y)
            #scipy.io.savemat(self.result_mat,mdict={'result':result})
        elif self.test_id==1:
            if not os.path.exists('result/'+self.dl_id):
                os.mkdir('result/'+self.dl_id)
            # train
            pre =self.result_mat[:-4]
            self.p_data['data_id'] =2
            if self.train_id>=9:
                self.p_data['data'] = 'dn_ucbg1.mat'
            else:
                self.p_data['data'] = 'dn_ucb1.mat'
            for i in range(1,10):
                print "do: image "+str(i)
                self.p_data['im_id'] = i
                self.loadData(self.path_test,'test')
                result = self.runTest(metric=-1)
                scipy.io.savemat(pre+'_'+str(i)+'.mat',mdict={'result':result})
        elif self.test_id==2:
            if not os.path.exists('result/'+self.dl_id):
                os.mkdir('result/'+self.dl_id)
            # test
            self.p_data['data_id'] =2
            if self.train_id>=9:
                self.p_data['data'] = 'dn_ucbg2.mat'
            else:
                self.p_data['data'] = 'dn_ucb2.mat'
            pre =self.result_mat[:-4]
            for i in range(1,10):
                print "do: image "+str(i)
                self.p_data['im_id'] = i
                self.loadData(self.path_test,'test')
                result = self.runTest()
                scipy.io.savemat(pre+'_'+str(i)+'.mat',mdict={'result':result})
            #scipy.io.savemat(self.result_mat,mdict={'done':1})
        elif self.test_id==3:
            self.loadData(self.path_test,'test',options={'data_id':3,'data':'test_im.mat'})
            result = self.runTest(metric=0)
 
    def buildModel(self):
        num_in = np.prod(self.ishape.shape)*self.ishape.num_channels
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
                [self.param.param_model_fc(dim = self.num_dim[0],irange=0.06,layer_type=1)],
                [self.param.param_model_cf(n_classes = self.num_dim[1],irange=0.07,layer_type=0)]
                ]
        elif self.model_id ==1:        
             # 2 tanh + 1 softmax
            self.p_layers = [
                [self.param.param_model_fc(dim = self.num_dim[0],irange=0.05,layer_type=1),
                 self.param.param_model_fc(dim = self.num_dim[1],irange=0.05,layer_type=1)],
                [self.param.param_model_cf(n_classes = self.num_dim[2],irange=0.05,layer_type=0)]
                ]
        elif self.model_id ==2:        
             # 2 tanh + 1 softmax
            self.p_layers = [
                [self.param.param_model_fc(dim = self.num_dim[0],irange=0.05,layer_type=1),
                 self.param.param_model_fc(dim = self.num_dim[1],irange=0.05,layer_type=1),
                 self.param.param_model_fc(dim = self.num_dim[2],irange=0.05,layer_type=1)],
                [self.param.param_model_cf(n_classes = self.num_dim[3],irange=0.05,layer_type=0)]
                ]
        elif self.model_id ==3:        
            # 1 tanh +1 linear
            n1 = np.sqrt(6.0/(num_in+num_dim[0]))
            n2 = np.sqrt(6.0/(num_dim[1]+num_dim[0]))
            #print "init:",n1,n2
            self.p_layers = [
                    [self.param.param_model_fc(dim = self.num_dim[0],irange=n1,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=n2,layer_type=2)]
                    ]

        elif self.model_id ==4:        
            # 1 tanh +1 linear
            n1 = np.sqrt(6.0/(num_in+num_dim[0]))
            n2 = np.sqrt(6.0/(num_dim[1]+num_dim[0]))
            n3 = np.sqrt(6.0/(num_dim[1]+num_dim[2]))
            self.p_layers = [
                    [self.param.param_model_fc(dim = self.num_dim[0],irange=n1,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=n2,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=n3,layer_type=2)]
                    ]

        elif self.model_id ==5:        
            # 1 tanh +1 linear
            n1 = np.sqrt(6.0/(num_in+num_dim[0]))
            n2 = np.sqrt(6.0/(num_dim[1]+num_dim[0]))
            n3 = np.sqrt(6.0/(num_dim[1]+num_dim[2]))
            n4 = np.sqrt(6.0/(num_dim[3]+num_dim[2]))
            self.p_layers = [
                    [self.param.param_model_fc(dim = self.num_dim[0],irange=n1,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[1],irange=n2,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[2],irange=n3,layer_type=1),
                    self.param.param_model_fc(dim = self.num_dim[3],irange=n4,layer_type=2)]
                    ]
        elif self.model_id ==6:
            # 1 conv + 1 tanh + 1 linear
            # feature extraction + regression
            nk = [self.num_dim[0],30,20]
            ks = [[8,8],[5,5],[3,3]]
            ir = [0.05,0.05,0.05]
            ps = [[1,1],[4,4],[2,2]]
            pd = [[1,1],[2,2],[2,2]]
            crop_len = [(1+(self.ishape.shape[k]-ks[0][k])/pd[0][k])/ps[0][k]  for k in [0,1] ]
            crop_cen = [(self.ishape.shape[k]-crop_len[k])/2 for k in [0,1] ]
            #print crop_len,crop_cen
            self.p_data['crop_y'] = U_centerind(self.ishape.shape,crop_cen,crop_len)
            #print self.p_data['crop_y'].size,self.p_data['crop_y'] 
            self.p_layers = [
                [self.param.param_model_conv(nk[0],ks[0],ps[0],pd[0],ir[0],layer_type=0)],
                [self.param.param_model_fc(dim = self.num_dim[1],irange=0.1,layer_type=1),
                self.param.param_model_fc(dim = self.num_dim[2],irange=0.1,layer_type=2)]
                    ]
    def buildAlgo(self):
        if self.algo_id == 0:
            algo_lr = 1e-4
            algo_mom = 1e-3
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

    def test_sketchtoken(self,yhat,yy=None):
        if yy==None:
            yy = self.DataLoader.data['test'].y
        if yy.shape[1]>1:
            yy = np.argmax(yy,axis=1)
        if yhat.shape[1]>1:
            yhat = np.argmax(yhat,axis=1)
        yhat[yhat!=150] = 0
        yy[yy!=150] = 0
        #print yhat.shape,yy.shape
        print float(sum(yhat!=yy))/len(yy)
    
"""
# linear layer: test data/cost function
python P_occ.py 0 -1 0 151 0
# softmax layer: test data/cost function
python P_occ.py 0 -2 100 151 0
# test model 1
python P_occ.py 0 1 1000 1000,289 0
# test model 3
python P_occ.py 0 3 1000 500,500,289 0

python P_occ.py 0 5 2000 500,500,500,289 4 3

"""


if __name__ == "__main__":             
    if len(sys.argv) != 7:
        raise('need six inputs: algo_id, model_id epoch_num layer train_id test_id')
    num_dim = sys.argv[4].split(',')
    num_dim = [int(x) for x in num_dim]
    exp = Deep_occ(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),num_dim,int(sys.argv[5]),int(sys.argv[6]))
    exp.run()
    """
    import cPickle
    nn = 'dl_p1_1_6.pkl' 
    a=cPickle.load(open(nn))
    """
    
