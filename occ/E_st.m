addpath('../util')
init
if ~exist('Is','var')
    load data/test/dn_ucb2
end
num_I = numel(Is);
sts = cell(1,num_I);
Es = cell(1,num_I);
errs = cell(1,num_I);
ntree = 2;
if ~exist('model','var')
    load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelSmall.mat']);
    if ntree
        model = U_st1(model);
    end
end

addpath([VLIB 'Mid/Boundary/SketchTokens'])
addpath(genpath([VLIB '../Piotr']))
%for i=1:1%num_I
parfor i=1:num_I
    st = stDetect( Is{i}, model );
    [~,sts{i}] = max(st,[],3);
    Es{i} = stToEdges( st, 1 ); 
    if mod(i,10)==0
        disp(i)
    end
end
if ntree
    save(['st_bd_' num2str(ntree)],'sts','Es')
else
    save st_bd sts Es
end
