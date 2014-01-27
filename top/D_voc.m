addpath('../util')
init

addpath([DATA_VOC '/VOCcode'])
VOCinit

imgset='train';
ids=textread(sprintf(VOCopts.seg.imgsetpath,imgset),'%s');


cl_id = 10;
num_train = numel(ids);
imgset = zeros(1,num_train);
parfor i= 1:num_train
    cls=imread(sprintf(VOCopts.seg.clsimgpath,ids{i}));
    imgset(i)= nnz(cls==cl_id)>0;
end
opt.cl_id = cl_id;
opt.CROP_DIM = 256;
opt.BD_DIM = 10;
opt.Corner = 0.5;
for id= find(imgset)
    im = imread(sprintf(VOCopts.imgpath,ids{id}));
    opt.seg = imread(sprintf(VOCopts.seg.clsimgpath,ids{id}));
    for j =1:3
        opt.id = j;
        imwrite(U_bb(im,opt),[num2str(cl_id) '_' num2str(j) '_' ids{id} '.png']);
    end 
end
