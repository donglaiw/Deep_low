%load spam0

param.K=256;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=1; % number of threads
param.batchsize = 300;
param.verbose=false;
param.iter=1000;

%{
param.mode =0; % sparse in coeff
param.gamma1=0.3;% sparse in dict
%}
% omp:
param.mode = 3; % sparse in coeff
param.lambda = 10;% sparse in dict

param.modeD=0;
param.modeParam=0; % opt
param.iter=1000;  % let us see what happens after 1000 iterations.

%%%%%%%%%% FIRST Emat_xPERIMENT %%%%%%%%%%%
addpath(genpath('/home/Stephen/Desktop/VisionLib/Donglai/Opt/spams-matlab'))
start_spams;

%{
% 
ind= reshape(1:35^2,[35 35]);
ind = ind(3:end-2,3:end-2);

%}
D = mexTrainDL(double(mat_x(ind,:)),param);
%D = mexTrainDL(double(mat_x),param);



%alpha=mexLasso(mat_x,D,param);
ImD=displayPatches(D);
imwrite(ImD,'dic_p.png')
save sc_D D


%{
kmeans:

addpath(genpath('/home/Stephen/Desktop/VisionLib/Donglai/Mid/Boundary/SketchTokens'))
load('models/forest/modelSmall.mat');
D=double(reshape(model.clusters,[],150));
D=D./repmat(sqrt(sum(D.^2)),[size(D,1) 1]);
save kmeans_D D


%}
