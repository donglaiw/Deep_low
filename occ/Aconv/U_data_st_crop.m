load('../data/train/train_imc')
load('../data/train/train_im')

ind = reshape(1:3675,[35 35 3]);
ind2 = reshape(ind(11:25,11:25,:),1,[]);

mat_x = uint8(train_imc(ind2,:));
mat_y = uint8(train_im(1,:));
save('../data/train/ucb_st_15','mat_x','mat_y')
