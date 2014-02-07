addpath('../../util'),init
%load /data/vision/billf/manifold-learning/DL/Data/BSD/dn_ucb2

rr=cell(1,10);
for id = 2:10;

test_id =1;
I = Is{id};
sz = size(I);
psz=11;
psz_h = (1+psz)/2;
nn = sprintf('result/0_0_0_96_1_0_1000/dl_r%d_%d.mat',test_id,id-1);
        if exist(nn,'file')
            load(nn) 
            yhat = zeros(sz(1:2));
            yhat(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1)) = reshape(result{1},sz([2 1])-psz+1)';
            %yhat=scol2im(result{1}',psz,sz(1),sz(2),'average');
        end        
        %yhat(yhat<0)=0;
rr{id}=  yhat;
end
save tmp_r rr

