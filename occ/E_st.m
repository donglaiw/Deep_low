addpath('../util')
init
if ~exist('Is','var')
    load data/test/dn_ucb
end
num_I = numel(Is);
sts = cell(1,num_I);
Es = cell(1,num_I);
errs = cell(1,num_I);
if ~exist('model','var')
    load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelSmall.mat']);
e
nd
addpath([VLIB 'Mid/Boundary/SketchTokens'])
addpath(genpath([VLIB '../Piotr']))
for i=1:num_I
st = stDetect( Is{i}, model );
[~,sts{i}] = max(st,[],3);
Es{i} = stToEdges( st, 1 ); 
end
save st_bd sts Es

