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
        self.path_train = '../data/train/'
        self.path_test = '../data/test/'            
        self.p_data = {'ds_id':0}   # occ data         
        self.batch_size = 100
        if self.train_id <= 2:
            self.p_data['data_id'] = 6
            if self.model_id<=1:
                self.psz = 11
            elif self.model_id<=4:
                self.psz = 15
            elif self.model_id<=5:
                self.psz = 17
            elif self.model_id<=6:
                self.psz = 15
            self.ishape = Conv2DSpace(shape = (self.psz,self.psz),num_channels = 3)

            if (self.model_id==6 and self.num_dim[-1]==4) or (self.model_id==3 and self.num_dim[-1]==4) or (self.model_id in [2,5] and self.num_dim[-1]==1):
                self.p_data['pre_id'] = 1
                self.p_data['im_id'] = 1
        elif self.train_id <= 4:
            # contour completion
            if self.train_id==3:
                self.p_data['data_id'] = 7
            elif self.train_id==4:
                self.p_data['data_id'] = 8
            self.psz = 17
            self.ishape = Conv2DSpace(shape = (self.psz,self.psz),num_channels = 1)
        elif self.train_id == 5:
            # pb + cnn 
            self.p_data['data_id'] = 10
            self.psz = 15
            self.ishape = Conv2DSpace(shape = (self.psz,self.psz),num_channels = 1)


        self.p_data['ishape']= np.append(self.ishape.shape,self.ishape.num_channels)


    def loadData_train(self):        
        #train_id = range(1,31000,3)
        if self.train_id==0:
            num_im = 200000
            valid_set = range(0,num_im,10)
            train_set = list(set(range(0,num_im)).difference(set(self.valid_set)))
            self.loadData(self.path_train,'train',train_set)
            self.loadData(self.path_train,'valid',valid_set)
        elif self.train_id==1:
            self.p_data['data']='conv_'+str(self.psz)+'_0.mat'
            self.loadData(self.path_train,'train')
            self.p_data['data']='conv_'+str(self.psz)+'_1.mat'
            self.loadData(self.path_train,'valid')
        elif self.train_id==2:
            self.nump = 1000
            self.p_data['data']=['ucb_0_'+str(self.psz)+'_2_1_'+str(self.nump)+'.mat','ucb_0_'+str(self.psz)+'_2_3_'+str(self.nump)+'.mat']
            if self.psz==11:
                del self.p_data['data'][0]
            self.loadData(self.path_train,'train')
            if self.DataLoader.data['train'].X.dtype==np.uint8:
                self.DataLoader.data['train'].X = self.DataLoader.data['train'].X.astype('float32')/255
            self.p_data['data']=['ucb_1_'+str(self.psz)+'_2_1_'+str(self.nump)+'.mat','ucb_1_'+str(self.psz)+'_2_3_'+str(self.nump)+'.mat']
            if self.psz==11:
                del self.p_data['data'][0]
            self.loadData(self.path_train,'valid')
            if self.DataLoader.data['train'].X.dtype==np.uint8:
                self.DataLoader.data['valid'].X = self.DataLoader.data['valid'].X.astype('float32')/255
            print "yoyo:",self.DataLoader.data['valid'].X.shape
        elif self.train_id==3:
            self.p_data['data']='mlp_st_0x.bin'
            self.loadData(self.path_train,'train')
            self.p_data['data']='mlp_st_1x.bin'
            self.loadData(self.path_train,'valid')
        elif self.train_id==4:
            self.p_data['data']='mlp_st_0sx.bin'
            self.loadData(self.path_train,'train')
            self.p_data['data']='mlp_st_1sx.bin'
            self.loadData(self.path_train,'valid')
        elif self.train_id==5:
            self.nump = 1000
            self.p_data['data']=['ucb_0_'+str(self.psz)+'_3_1_'+str(self.nump)+'.mat','ucb_0_'+str(self.psz)+'_3_3_'+str(self.nump)+'.mat']
            self.loadData(self.path_train,'train')
            self.p_data['data']=['ucb_1_'+str(self.psz)+'_3_1_'+str(self.nump)+'.mat','ucb_1_'+str(self.psz)+'_3_3_'+str(self.nump)+'.mat']
            self.loadData(self.path_train,'valid')

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
        #self.loadWeight(self.param_pkl)
        if self.test_id==-2:
            self.p_data['data'] = 'db_11.mat'
            self.p_data['data_id'] = 6
            self.loadData('./','train')
            result = self.runTest(self.DataLoader.data['train'],2)
            #print result[0]
            #scipy.io.savemat('db.mat',mdict={'result':result})
           #single mat
        elif self.test_id==0:
            self.p_data['data']=['ucb_0_'+str(self.psz)+'_2_1.mat','ucb_0_'+str(self.psz)+'_2_3.mat']
            self.loadData(self.path_train,'train')
            self.p_data['data']=['ucb_1_'+str(self.psz)+'_2_1.mat','ucb_1_'+str(self.psz)+'_2_3.mat']
            self.loadData(self.path_train,'valid')

            result = self.runTest(self.DataLoader.data['train'],2)
            result2 = self.runTest(self.DataLoader.data['valid'],2)
            scipy.io.savemat('ha.mat',mdict={'r1':result,'r2':result2})
        elif self.test_id<=4:
            #self.path_test = '../data/train/'
            #self.path_test = '../data/test/'+str(self.psz)+'/'
            self.path_test = '../data/test/'
            # e.g.  python P_occ_conv.py 0 0 1000 0,96,1 2 1
            if not os.path.exists('result/'+self.dl_id):
                os.mkdir('result/'+self.dl_id)
            # train
            pre =self.result_mat[:-5]
            self.p_data['data_id'] = 11
            self.p_data['data'] = ['bd_st2.mat']
            #self.p_data['data_id'] = 9
            #self.p_data['data'] = ['dn_ucb0.mat']
            tid = self.test_id-1
            for i in range(50*tid,50*(tid+1)):
                print "do: image "+str(i)
                self.p_data['im_id'] = i
                self.loadData(self.path_test,'test')
                #self.DataLoader.data['test'].X = self.DataLoader.data['test'].X.astype('float32')/255
                result = self.runTest(metric=-1)
                scipy.io.savemat(pre+'_'+str(i)+'.mat',mdict={'result':result})

    def buildModel(self):
        num_in = np.prod(self.ishape.shape)*self.ishape.num_channels
        if self.model_id == -2:
            # python P_occ_conv.py 0 -2 1000 81 4 0            
            n1 = 0.005
            self.p_layers = [
                [self.param.param_model_fc(dim = self.num_dim[0],irange=n1,layer_type=1)]
                ]

        elif self.model_id == -1:
            # 1 tanh + 1 softmax            
            # python P_occ_conv.py 0 -1 1000 1,1 4 0
            ks = [[11,11],[9,9],[3,3]]
            ir = [0.01,0.01,0.01]
            ps = [[1,1],[1,1],[2,2]]
            pd = [[1,1],[1,1],[2,2]]
            kid = self.num_dim[0]
            n1 = 0.03
            crop_len = [(1+(self.ishape.shape[k]-ks[kid][k])/pd[kid][k])/ps[kid][k]  for k in [0,1] ]
            #print self.ishape.shape,ks[kid]
            if max(crop_len)>1:
                crop_cen = [(self.ishape.shape[k]-crop_len[k])/2 for k in [0,1] ]
                #print crop_len,crop_cen
                self.p_data['crop_y'] = U_centerind(self.ishape.shape,crop_cen,crop_len)
                #print self.p_data['crop_y'].size,self.p_data['crop_y'] 
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[1],ks[kid],ps[kid],pd[kid],ir[kid],layer_type=4)]
                ]

        elif self.model_id ==0:
            # 1 tanh + 1 softmax            
            ks = [[11,11],[9,9],[3,3]]
            ir = [0.5,0.01,0.01]
            ps = [[1,1],[1,1],[2,2]]
            pd = [[1,1],[1,1],[2,2]]
            kid = self.num_dim[0]
            n1 = 0.03
            crop_len = [(1+(self.ishape.shape[k]-ks[kid][k])/pd[kid][k])/ps[kid][k]  for k in [0,1] ]
            #print self.ishape.shape,ks[kid]
            if max(crop_len)>1:
                crop_cen = [(self.ishape.shape[k]-crop_len[k])/2 for k in [0,1] ]
                #print crop_len,crop_cen
                self.p_data['crop_y'] = U_centerind(self.ishape.shape,crop_cen,crop_len)
                #print self.p_data['crop_y'].size,self.p_data['crop_y'] 
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[1],ks[kid],ps[kid],pd[kid],ir[kid],layer_type=0)],
                [self.param.param_model_fc(dim = self.num_dim[2],irange=n1,layer_type=2)]
                ]
        elif self.model_id ==1:
            # 1 tanh + 1 softmax            
            ks = [[11,11],[1,1],[3,3]]
            ir = [0.01,0.01,0.01]
            ps = [[1,1],[1,1],[2,2]]
            pd = [[1,1],[1,1],[2,2]]
            kid = self.num_dim[0]
            crop_len = [(1+(self.ishape.shape[k]-ks[kid][k])/pd[kid][k])/ps[kid][k]  for k in [0,1] ]
            if max(crop_len)>1:
                crop_cen = [(self.ishape.shape[k]-crop_len[k])/2 for k in [0,1] ]
                #print crop_len,crop_cen
                self.p_data['crop_y'] = U_centerind(self.ishape.shape,crop_cen,crop_len)
                #print self.p_data['crop_y'].size,self.p_data['crop_y'] 
            n1 = 0.01
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[1],ks[kid],ps[kid],pd[kid],ir[kid],layer_type=0)],
                [self.param.param_model_fc(dim = self.num_dim[2],irange=n1,layer_type=1),
                self.param.param_model_fc(dim = self.num_dim[3],irange=n1,layer_type=2)]
                ]
        elif self.model_id ==2:        
            ks = [[11,11],[5,5]]
            ir = [0.01,0.01]
            ps = [[1,1],[1,1]]
            pd = [[1,1],[1,1]]
            n1 = 0.01
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[0],ks[0],ps[0],pd[0],ir[0],layer_type=0),
                self.param.param_model_conv(self.num_dim[1],ks[1],ps[1],pd[1],ir[1],layer_type=0)],
                [self.param.param_model_fc(dim = self.num_dim[2],irange=n1,layer_type=self.num_dim[3])]
                ]

        elif self.model_id ==3:        
            ks = [[11,11],[5,5],[1,1]]
            ir = [0.01,0.01,0.01]
            ps = [[1,1],[1,1],[1,1]]
            pd = [[1,1],[1,1],[1,1]]
            n1 = 0.01
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[0],ks[0],ps[0],pd[0],ir[0],layer_type=0),
                self.param.param_model_conv(self.num_dim[1],ks[1],ps[1],pd[1],ir[1],layer_type=0),
                self.param.param_model_conv(self.num_dim[2],ks[2],ps[2],pd[2],ir[2],layer_type=self.num_dim[3])]
                ]
        elif self.model_id ==4:        
            ks = [[11,11],[5,5],[1,1],[1,1]]
            ir = [0.01,0.01,0.01,0.01]
            ps = [[1,1],[1,1],[1,1],[1,1]]
            pd = [[1,1],[1,1],[1,1],[1,1]]
            n1 = 0.01
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[0],ks[0],ps[0],pd[0],ir[0],layer_type=0),
                self.param.param_model_conv(self.num_dim[1],ks[1],ps[1],pd[1],ir[1],layer_type=0),
                self.param.param_model_conv(self.num_dim[2],ks[2],ps[2],pd[2],ir[2],layer_type=2),
                self.param.param_model_conv(self.num_dim[3],ks[3],ps[3],pd[3],ir[3],layer_type=2)]
                ]

        elif self.model_id ==5:        
            # man... initialization matters
            # ratio between the learning rate
            ks = [[11,11],[5,5],[3,3]]
            ir = [1,1,1]
            ps = [[1,1],[1,1],[1,1]]
            pd = [[1,1],[1,1],[1,1]]
            n1 = 0.01
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[0],ks[0],ps[0],pd[0],ir[0],layer_type=0),
                self.param.param_model_conv(self.num_dim[1],ks[1],ps[1],pd[1],ir[1],layer_type=0),
                self.param.param_model_conv(self.num_dim[2],ks[2],ps[2],pd[2],ir[2],layer_type=0)],
                [self.param.param_model_fc(dim = self.num_dim[3],irange=n1,layer_type=self.num_dim[4])]
                ]
        elif self.model_id ==6:        
            ks = [[11,11],[5,5],[1,1],[1,1],[1,1]]
            ir = [0.01,0.01,0.01,0.01,0.01]
            ps = [[1,1],[1,1],[1,1],[1,1],[1,1]]
            pd = [[1,1],[1,1],[1,1],[1,1],[1,1]]
            n1 = 0.01
            self.p_layers = [
                [self.param.param_model_conv(self.num_dim[0],ks[0],ps[0],pd[0],ir[0],layer_type=0),
                self.param.param_model_conv(self.num_dim[1],ks[1],ps[1],pd[1],ir[1],layer_type=0),
                self.param.param_model_conv(self.num_dim[2],ks[2],ps[2],pd[2],ir[2],layer_type=self.num_dim[5]),
                self.param.param_model_conv(self.num_dim[3],ks[3],ps[3],pd[3],ir[3],layer_type=self.num_dim[5]),
                self.param.param_model_conv(self.num_dim[4],ks[4],ps[4],pd[4],ir[4],layer_type=self.num_dim[5])]
                ]


    def buildAlgo(self):
        if self.algo_id == 0:
            algo_lr = 1e-2
            algo_mom = 1e-1

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
        #print float(sum(yhat!=yy))/len(yy)
    
"""
# linear layer: test data/cost function
python P_occ_conv.py 0 0 1 0 0 0

import cPickle;a=cPickle.load(open('dl_p0_1_0_0_3.pkl'))
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
    
