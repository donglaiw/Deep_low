addpath('../../util'),init
if ~exist('Is','var')
    load /data/vision/billf/manifold-learning/DL/Data/BSD/dn_ucb2
end

if ~exist('Iss','var')
    load /data/vision/billf/manifold-learning/DL/Data/BSD/dn_ucb_Is2
end
pb=cell(1,200);
test_id =1;
psz=15;
psz_h = (1+psz)/2;
do_pre = 0;

do_s=3;
parfor id = 1:200;
sz = size(Is{id});
switch do_s
case 1
sz = size(Iss{id}{1});
case 2
sz = size(Iss{id}{1});
sz2 = size(Iss{id}{2});
end
%nn = sprintf('result/0_3_50_50_1_4_2_63/dl_r_%d.mat',id-1);
%nn = sprintf('result/0_3_20_20_1_4_2_115/dl_r%d_%d.mat',test_id,id-1);
%nn = sprintf('result/0_2_20_20_1_2_1000/dl_r%d_%d.mat',test_id,id-1);
%nn = sprintf('result/0_0_0_96_1_0_1000/dl_r%d_%d.mat',test_id,id-1);
%nn = sprintf('result/0_0_0_96_1_2_1000/dl_r%d_%d.mat',test_id,id-1);
%nn = sprintf('result/0_3_20_20_1_4_5_200/dl_r_%d.mat',id-1);
%nn = sprintf('result/0_11_50_50_1_2_4_2_100_0/dl_r_%d.mat',id-1);
%nn = sprintf('result/0_11_50_50_1_2_4_2_100/dl_r_%d.mat',id-1);
%nn = sprintf('result/0_11_50_50_1_2_4_9_100_0/dl_r_%d.mat',id-1);
nn = sprintf('result/0_13_50_50_151_2_0_200/dl_r_%d.mat',id-1);
            tmp=load(nn) 
           if do_s~=2 
            yhat = zeros(sz(1:2),'single');
        else
            yhat = zeros(sz2(1:2),'single');
        end
            if do_pre
                yhat(:) = -1;
            end

        switch do_s
        case 0
            yhat(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1)) = reshape(tmp.result{1},sz([2 1])-psz+1)';
        case 1
            yhat(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1)) = reshape(tmp.result{1}(1:prod(sz(1:2)-psz+1)),sz([2 1])-psz+1)';
        case 2
            yhat(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1)) = reshape(tmp.result{1}((1+prod(sz(1:2)-psz+1)):end),sz2([2 1])-psz+1)';
        case 3
            yhat(psz_h:(end-psz_h+1),psz_h:(end-psz_h+1)) = reshape(1-tmp.result{1}(:,end),sz([2 1])-psz+1)';
        end
            if do_pre
                yhat = yhat/2+0.5;
            end
            %yhat=scol2im(result{1}',psz,sz(1),sz(2),'average');
        %yhat(yhat<0)=0;
pb{id}=  yhat;
end
save tmp_r_50_50_1 pb
%parfor i=1:200;pb{i}=single(pb{i});end
