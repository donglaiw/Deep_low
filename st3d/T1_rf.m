function T1_rf(id)
addpath('../util');init
addpath(genpath([VLIB 'Mid/Boundary/SketchTokens/']))
addpath(genpath([VLIB 'Util/io']))
addpath(genpath([VLIB '../Piotr']))

opts=struct('DD',STRACK_DIR,...
            'loadmat','segtrack',...
            'radius',17,...
            'pratio',0.3,...
            'tsz',  5,...
            'tstep', 2,...
            'num_pervol',100,...
            'modelFnm','model3D');
%st3dTrain(opts);
try 
    matlabpool
end
    st3dTrain_p(opts,id);
%{
addpath('../util');init
PP=pwd;
system([VLIB 'Para/p_run.sh 1 1 2 14 "' PP '" "T1_rf(" ");" "' PP '/sc" ".mat"'])
%}
