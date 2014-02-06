function model = U_st1(model)
ntree = 1;
model.opts.nTrees = 2;
model.thrs(:,(ntree+1):end) = [];
model.fids(:,(ntree+1):end) = [];
model.child(:,(ntree+1):end) = [];
model.count(:,(ntree+1):end) = [];
model.depth(:,(ntree+1):end) = [];
model.distr(:,:,(ntree+1):end) = [];

model.thrs(:,2) = model.thrs(:,1);
model.fids(:,2) = model.fids(:,1);
model.child(:,2) = model.child(:,1);
model.count(:,2) = model.count(:,1);
model.depth(:,2) = model.depth(:,1);
model.distr(:,:,2) = model.distr(:,:,1);

