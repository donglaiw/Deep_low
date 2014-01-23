





import os
#train_id = ['9','10']
#test_id = ['1','2']
train_id = ['9']
test_id = ['0']
dim_n = ['151','2']

m_id = ['500','500,500','500,500,500']
m_i = ['0','1','2']
num_epoch = '100'
for te_id in test_id:
    for j in range(len(train_id)):
        tr_id = train_id[j]
        dn = dim_n[j]
        for i in range(len(m_id)):
            nn = ['1', m_i[i],m_id[i],dn,tr_id,num_epoch] 
            if True or not os.path.exists('result/'+'_'.join(nn)+'/dl_r'+te_id+'_9.mat'):
                print 'python P_occ.py 1 '+m_i[i]+' 100 '+m_id[i]+','+dn+' '+tr_id+' '+te_id
                a=os.popen('python P_occ.py 1 '+m_i[i]+' 100 '+m_id[i]+','+dn+' '+tr_id+' '+te_id)
