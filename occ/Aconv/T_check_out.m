addpath('../../util'),init
if ~exist('Is','var')
    load /data/vision/billf/manifold-learning/DL/Data/BSD/dn_ucb2
end

pb=cell(1,10);
test_id =1;
psz=15;
psz_h = (1+psz)/2;
do_pre = 0;
parfor id = 1:200;

I = Is{id};
sz = size(I);
%nn = sprintf('result/0_3_50_50_1_4_2_63/dl_r_%d.mat',id-1);
%nn = sprintf('result/0_3_20_20_1_4_2_115/dl_r%d_%d.mat',test_id,id-1);
%nn = sprintf('result/0_2_20_20_1_2_1000/dl_r%d_%d.mat',test_id,id-1);
%nn = sprintf('result/0_0_0_96_1_0_1000/dl_r%d_%d.mat',test_id,id-1);
%nn = sprintf('result/0_0_0_96_1_2_1000/dl_r%d_%d.mat',test_id,id-1);
nn = sprintf('result/0_3_20_20_1_4_5_200/dl_r_%d.mat',id-1);
            tmp=load(nn) 
            
            yhat = zeros(sz(1:2));
            if do_pre
                yhat(:) = -1;
            end
            yhat(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1)) = reshape(tmp.result{1},sz([2 1])-psz+1)';
            if do_pre
                yhat = yhat/2+0.5;
            end
            %yhat=scol2im(result{1}',psz,sz(1),sz(2),'average');
        %yhat(yhat<0)=0;
pb{id}=  yhat;
end
save tmp_r_50_50_1 pb
