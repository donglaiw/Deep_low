addpath('../util'),init
load([VLIB 'DeepL/Deep_wrapper/decaf/st_bd_2.mat'])


if ~exist('Is','var')
    load ../data/test/dn_ucb2
end
if ~exist('model','var')
    load([VLIB 'Mid/Boundary/SketchTokens/models/forest/modelFull.mat']);
end
addpath([VLIB 'Mid/Boundary/SketchTokens'])
addpath(genpath([VLIB '../Piotr']))


i = 10;
st = stDetect( Is{i}, model );
Es = stToEdges( st, 1 );
subplot(211),imagesc(Es)
subplot(212),imagesc(1-st(:,:,end))

addpath([VLIB 'Low/Filter/im_guide'])
pb =cell(1,4);


r = 4; % try r=2, 4, or 8
eps = 0.2^2; % try eps=0.1^2, 0.2^2, 0.4^2

pb{1} = guidedfilter_color(double(Is{i})/255, 1-st(:,:,end), r, eps);
subplot(212),imagesc(pb{1})


rs= [2 4 8];
epss= [0.1^2, 0.2^2, 0.4^2]
for i = 1:3
for j = 1:3
subplot(3,3,(i-1)*3+j),imagesc(guidedfilter_color(double(Is{i})/255, 1-st(:,:,end), rs(i), epss(j)))
end
end





% scale space
addpath([VLIB 'Low/Filter/matlabPyrTools'])
[a1,a2] = buildGpyr(1-double(st(:,:,end)),5,'gauss5');
ims=U_pyr2im(a1,a2);
opt =2;
for i=1:5
	switch opt
case 1
subplot(3,2,i),imagesc(imresize(ims{i},size(ims{1})))
case 2
tmp = single(imresize(ims{i},size(ims{1})));
subplot(3,2,i),imagesc(stToEdges(cat(3,tmp,1-tmp),1))
case 3
tmp = stToEdges(cat(3,single(ims{i}),1-single(ims{i})),1);
subplot(3,2,i),imagesc(imresize(tmp,size(ims{1})))
end
end

% sequential edgemap
im = 1-double(st(:,:,end));
sz= size(im);
for i=1:5
[a1,a2] = buildGpyr(double(im),2,'gauss5');
tmp = U_pyr2im(a1,a2);
im = stToEdges(cat(3,single(tmp{2}),1-single(tmp{2})),1);
%subplot(3,2,i),imagesc(imresize(im,sz))
subplot(3,2,i),imagesc(im)
im = tmp{2};
end

% sequential data-term
imss = cell(1,3);
for k=1:3
[a1,a2] = buildGpyr(double(Is{10}(:,:,k)),5,'gauss5');
imss{k}=U_pyr2im(a1,a2);
end
ims = cell(1,numel(imss{1}));
for k=1:numel(imss{1})
	ims{k} = cat(3,imss{1}{k},imss{2}{k},imss{3}{k})/(2^(k-1));
end



for i=1:5
st = stDetect( single(ims{i})/255, model );
subplot(3,2,i),imagesc(stToEdges( st, 1 ))
end
