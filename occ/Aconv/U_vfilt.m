nn = 'dl_p0_8_10_10_20_1_6_100.pkl';
%nn = 'dl_p0_0_0_20_1_0_1000.pkl';
%nn = 'dl_p0_0_0_96_1_0_1000.pkl';
%nn = 'dl_p0_1_0_96_100_1_0_1000.pkl';
%nn = 'dl_p0_1_0_96_1_2_1000.pkl';
system(['python U_p2m.py ' nn ' ha.mat'])
load ha.mat

psz = 11;
filt = reshape(permute(result{1,1},[4 3 2 1]),psz^2*3,[]);
%filt = reshape(permute(result{1,1},[3 4 2 1]),psz^2*3,[]);
addpath('/data/vision/billf/stereo-vision/VisionLib/Donglai/Opt/spams-matlab/build/')


tmp = filt(1:psz^2,:);
tmp = tmp-min(tmp(:));
tmp = tmp/max(tmp(:));
ImD=displayPatches(tmp);
%ImD=displayPatches(filt(1:psz^2,:));
