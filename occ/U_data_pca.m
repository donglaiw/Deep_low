ndim = 3000;
% scale to 0-1
fid = fopen('data/test/st_test.bin');
data = fread(fid, [7000 inf], 'single');
fclose(fid);
lb = data(:,1);
data = data(:,2:end);
min_d = min(data);
data = bsxfun(@minus,data,min_d);
max_d = max(data);
data = bsxfun(@rdivide,data,max_d);
tmp_cov = cov(data);
[aa,bb]= eigs(tmp_cov,ndim);

test_feat_s = [lb data*aa];
save test_feat_s test_feat_s
save eigen_feat aa bb

fid = fopen('data/train/st_train.bin');
data = fread(fid, [31000 inf], 'single');
fclose(fid);
lb = data(:,1);
data = data(:,2:end);
data = bsxfun(@minus,data,min_d);
data = bsxfun(@rdivide,data,max_d);
train_feat_s = [lb data*aa]; 
save train_feat_s train_feat_s
