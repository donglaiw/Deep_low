if ~exist('Is','var')
    load data/test/dn_ucb
end
num_I = numel(Is);
sts = cell(1,num_I);
Es = cell(1,num_I);
errs = cell(1,num_I);
load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelSmall.mat']);
parfor i=1:num_I
sts{i} = stDetect( Is{i}, model );
Es{i} = stToEdges( sts{i}, 1 ); 
end
save st_bd sts Es

