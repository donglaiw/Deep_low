

param= cell(2);
num_filt = 1;
param{1,1} = squeeze(zeros([num_filt,3,11,11],'single'));
%param{1,1} = zeros([1,3,11,11],'single');
param{1,1}(1,:,6,:) = 1;
param{1,2} = 0;
param{2,1} = 1;
param{2,2} = 0;
save init_p0_0 param

