%nn = 'dl_p0_0_0_96_1_0_1000.pkl';
%nn = 'dl_p0_1_0_96_100_1_0_1000.pkl';
nn = 'dl_p0_1_0_96_1_2_1000.pkl';
system(['python U_p2m.py ' nn ' ha.mat'])
load ha.mat

psz = 11;
filt = reshape(permute(result{1,1},[4 3 2 1]),psz^2*3,[]);
%filt = reshape(permute(result{1,1},[3 4 2 1]),psz^2*3,[]);
addpath('/data/vision/billf/stereo-vision/VisionLib/Donglai/Opt/spams-matlab/build/')

ImD=displayPatches(filt(1:psz^2,:));
