





import os
#train_id = ['9','10']
#test_id = ['1','2']
train_id = ['0']
test_id = ['1','2']
algo_id = ['0']
dim_n = ['289']

m_id = ['1000','1000,1000','1000,1000,1000']
m_i = ['1','3','4']
num_epoch = '1000'
for al_id in algo_id:
    for te_id in test_id:
        for j in range(len(train_id)):
            tr_id = train_id[j]
            dn = dim_n[j]
            for i in range(len(m_id)):
                nn = [al_id, m_i[i]]+m_id[i].split(',')+[dn,tr_id,num_epoch] 
                if not os.path.exists('result/'+'_'.join(nn)+'/dl_r'+te_id+'_6.mat'):
                #if True or not os.path.exists('result/'+'_'.join(nn)+'/dl_r'+te_id+'_1.mat'):
                    #print 'result/'+'_'.join(nn)+'/dl_r'+te_id+'_1.mat'
                    #print 'python P_occ.py '+al_id+' '+m_i[i]+' '+num_epoch+' '+m_id[i]+','+dn+' '+tr_id+' '+te_id
                    a=os.popen('python P_dn.py '+al_id+' '+m_i[i]+' '+num_epoch+' '+m_id[i]+','+dn+' '+tr_id+' '+te_id)
