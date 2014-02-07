
function B_infer_p(id)
addpath('../../util');init
load /data/vision/billf/manifold-learning/DL/Data/BSD/dn_ucb_pb2
load ../bd_st2
addpath(genpath('/data/vision/billf/stereo-vision/VisionLib/Donglai/Opt/spams-matlab'))
start_spams;

psz = 31;
algo_id = 1;
ssz = 0;


rr=cell(2,2);
for gid = 1:2;
switch gid
case 1
    sub_im = pb{id}((ssz+1):end-ssz,(ssz+1):end-ssz);
case 2
    sub_im = gts_pb{id}((ssz+1):end-ssz,(ssz+1):end-ssz);
end
mat_x2 = im2col(sub_im,[psz psz]);

for Did = 1:2;
switch Did
case 1
    load sc_D
case 2
    load km_D
end
switch algo_id
case 1 
    param.L= 5; % not more than 10 non-zeros coefficients
    param.eps=0.001; % squared norm of the residual should be less than 0.1
    a2=mexOMP(double(mat_x2),D,param);
case 2
    param.lambda=0.15; % not more than 20 non-zeros coefficients
    param.mode=1;        % penalized formulation
    a2=mexLasso(double(mat_x2),D,param);
end

im2 = scol2im(D*a2,psz,size(sub_im,1),size(sub_im,2),'average');
%subplot(211),imagesc(sub_im)
%subplot(212),imagesc(im2)
imwrite(im2/max(im2(:)),sprintf('out_%d_%d_%d.png',id,Did,gid))
rr{gid,Did} = im2;
end
save(sprintf('sc%d',id),'rr')
end
%{
addpath('../../util');init
PP=pwd;
%system([VLIB 'Para/p_run.sh 2 0 1 200 "' PP '" "B_infer_p(" ");" "' PP '/sc" ".mat"'])
system([VLIB 'Para/p_run.sh 2 1 1 200 "' PP '" "B_infer_p(" ");" "' PP '/sc" ".mat"'])


sp1 = cell(1,200);
for i=1:200;try;load(sprintf('sc%d',i));sp1{i}=rr{1,1};end
save sp1 sp1

%}
