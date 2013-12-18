%{
% generate clean
VLIB ='/data/vision/billf/stereo-vision/VisionLib/Donglai/'; 
PP=pwd;
system([VLIB 'Para/p_run.sh 2 1 1 30 "' PP '" "D_voc(" ");" "voc/dn_voc_" ".mat"'])
% generate noise
sig = 25;
for i = 1:30
    load(['voc/dn_voc_' num2str(i)]);
    save(['voc/dn_voc_p' num2str(i)],'pss');
    npss = uint8(single(pss)+single(rand(size(pss))*sig));
    clear pss
    save(['voc/n_voc_p' num2str(i)],'npss');
    clear npss
end
%}
function D_voc(id)
% extract 30M patches
stream=RandStream('mrg32k3a','Seed',1);
set(stream,'Substream',1);
RandStream.setGlobalStream( stream );

DATA_VOC = '/data/vision/billf/manifold-learning/DL/Data/VOC2012/JPEGImages/';
imgIds=dir([DATA_VOC '*.jpg']);
imgIds={imgIds.name};

% extract commonly used options
patchSiz=17;
matlabpool(6)
num_perbatch = 500;
num_perimg = 2000;
ps =cell(1,num_perbatch);
preid  = (id-1)*num_perbatch;

parfor i = 1:num_perbatch
    I = imread([DATA_VOC imgIds{preid+i}]);    
    tmp = [im2col(I(:,:,1),[patchSiz patchSiz]);...
    im2col(I(:,:,2),[patchSiz patchSiz]);...
    im2col(I(:,:,3),[patchSiz patchSiz])];
    ps{i} = tmp(:,randsample(size(tmp,2),num_perimg));
end

pss = cell2mat(ps);
save(['voc/dn_voc_' num2str(id)],'pss');
