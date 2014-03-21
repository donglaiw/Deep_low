
DD='~/Desktop/Data/';



%  CMU:
fns = dir([DD 'Occ/CMU/clips']);
fns(1:2)=[];
DD2= '/home/Stephen/Desktop/Edge/Hack/result/';

% st-2D
st = cell(1,numel(fns));
for i=1:numel(fns)
load([DD2 'CMU_' fns(i).name '.mat'],'ims_bd');
fn  = dir([DD 'Occ/CMU/clips/' fns(i).name '/ground_truth*']);
id = str2double(fn.name(find(fn.name=='_',1,'last')+1:end-4))+1;
st{i} = ims_bd(:,:,id);
end

% st-3D

VLIB = '/home/Stephen/Desktop/VisionLib/Donglai/';
addpath(genpath([VLIB 'Util']))
addpath(genpath([VLIB 'Mid/Boundary/SketchTokens']))
addpath(genpath([VLIB '../Piotr']))
load model3D
tsz_h = 2;
st3d = cell(1,numel(fns));
parfor i=1:numel(fns)
im = uint8(U_fns2ims([DD 'Occ/CMU/clips/' fns(i).name '/img_']));
fn  = dir([DD 'Occ/CMU/clips/' fns(i).name '/ground_truth*']);
id = str2double(fn.name(find(fn.name=='_',1,'last')+1:end-4))+1;
tmp_st = st3dDetect( im(:,:,:,id+(-tsz_h:tsz_h)), model );
%ims_bd(:,:,k) = stToEdges( st, 1 );
st3d{i} =tmp_st(:,:,1);
end

for i=1:numel(fns)
    imwrite(st3d{i},['st3d_' fns(i).name '.png'])
    imwrite(st{i},['st_' fns(i).name '.png'])
end