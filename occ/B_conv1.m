algo_id = 1;

%{
train=load('data/train/conv_11_0');
valid=load('data/train/conv_11_1');
test=load('data/test/conv_11_2');
%}
%algo_id:
% 1: malik
f_train = U_conv_feat(train,algo_id);
ww = (f_train')\train.mat_y';
save(['B1_' num2str(algo_id)],'ww')
f_valid = U_conv_feat(valid,algo_id);
f_test = U_conv_feat(test,algo_id);
mean(((f_train')*ww-train.mat_y').^2)
mean(((f_valid')*ww-valid.mat_y').^2)
mean(((f_test')*ww-test.mat_y').^2)
% 1: 0.0055, 0.0056, 0.0060 
