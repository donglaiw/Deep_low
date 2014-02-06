% initial error
psz = 17;
psz2 = 9;
psz2_h = (1+psz2)/2;
psz_c = (1+psz^2)/2;
psz2_c = (1+psz2^2)/2;

psz_d = (psz-psz2)/2;
ind_mat = reshape(1:psz^2,[psz psz]);
ind_mat = ind_mat(psz_d+1:end-psz_d,psz_d+1:end-psz_d);
ind_mat = ind_mat(:);

tid =1;

addpath(genpath('/home/Stephen/Desktop/VisionLib/Piotr'))
did=2;
load('../bd_st2')
pb_out =cell(1,numel(pb));

for i=1:numel(pb)	
	switch tid
	case 2
		% L2 error		
			if ~exist('w','var')
				load('ww','w')
			end
			mat_x = im2col(imPad(pb{i},psz2_h-1,'replicate'),[psz2 psz2]);
			pb_out{i} = reshape(w'*mat_x,size(pb{i}));
			pb_out{i}(pb_out{i}<0)=0;pb_out{i}(pb_out{i}>1)=1;
			% 0.0055,0.0055
	end	
end




addpath('/home/Stephen/Desktop/VisionLib/Donglai/Mid/Boundary/Bd_eval')
load('/home/Stephen/Desktop/Data/Seg/BSR/dn_ucb2','gts')
num_test = 16;
pb2= pb_out;
parfor j=1:num_test;j,[roc(j),re(j,:,:)] = U_occ(pb2{j},gts{j});end
re_m = squeeze(mean(re,1));
r = re_m(:,2)./(re_m(:,1)+(re_m(:,1)==0));
p = re_m(:,4)./(re_m(:,3)+(re_m(:,3)==0));
bid = r+p==0; r(bid)=[]; p(bid)=[];
f = 2.*r.*p./(r+p+((r+p)==0));
[max(f),U_roc(r,p),mean(roc)]

load('/home/Stephen/Desktop/Data/Seg/BSR/dn_ucb_pb2')
% l2 error?
mean(cell2mat(arrayfun(@(x) reshape((gts_pb{x}-pb2{x}).^2,1,[]),1:200,'UniformOutput',false)))


pb_out2 = pb_out;
for i=1:numel(pb_out);pb_out2{i}=(pb_out2{i}-min(pb_out2{i}(:)))/(max(pb_out2{i}(:))-min(pb_out2{i}(:)));end
pb_out3 = pb_out2;
for i=1:numel(pb_out);pb_out3{i}=U_stToEdges(pb_out2{i},1,5);end
%pb:     0.6765    0.7106    0.7377  (0.0357)
%pb_out: 0.6526    0.6600    0.6707 (0.0057)
%pb_out2:0.6633    0.6802    0.6906 (0.0186)
%pb_out3:0.6672    0.6907    0.7012 (0.0120)
% nms



% check filters
addpath('/home/Stephen/Desktop/VisionLib/Donglai/Low/Filter/bfilter2')
id=33;

id=22;
sz = size(pb{id});
subplot(221),imagesc(pb{id})
subplot(222),imagesc(gts_pb{id})
subplot(223),tmp=U_maxpool(pb{id},4,3);imagesc(imresize(tmp,sz))
subplot(224),imagesc(imresize(U_maxpool(tmp,4,3),sz))
imagesc(U_bila(pb{id},pb{id},[3,3],[0.001,0.001]))


% upperbound
pb_out4 = pb;
for i=1:numel(pb);
	mask=imfilter(double(gts_pb{i}~=0),fspecial('gaussian',5,1));
	pb_out4{i}(mask==0)=0;
	end
% upperbound: 0.9573    0.9305    0.9395


% maxpool pyramid
tmp = pb{id};
sz = size(tmp);
subplot(3,2,1),imagesc(tmp)
for i=1:5
	tmp = U_maxpool(tmp,3,2); 
	subplot(3,2,i+1),imagesc(imresize(tmp,sz,'nearest'))
end

% gaussian pyramid
addpath('../../util'),init
addpath([VLIB 'Low/Flow/Deqing/utils'])
f = fspecial('gaussian', [5 5], 1.5);
%f = fspecial('sobel');
%f = fspecial('laplacian',1);
nL = 6;
ratio = 0.6;
ims=compute_image_pyramid(pb{id},f, nL, ratio);
for i=2:6
	subplot(3,2,i),imagesc(imresize(ims{i},sz,'nearest'))
end


% filterbank regression
addpath('~/Desktop/VisionLib/Donglai/DeepL/Deep_wrapper/decaf')
fsz=31;
ff= U_f_LM(fsz);
%{
for i=1:9
	subplot(3,3,i),imagesc(conv2(pb{id},ff(:,:,i)))
	end
%}	
	im2 = zeros([size(pb{id}) 48]);
for i=1:48
	im2(:,:,i) = conv2(pb{id},ff(:,:,i),'same');
end
im3 = im2;im3(im2<0)=0;
w=reshape(im3,[],48)\gts_pb{id}(:);
subplot(221),imagesc(reshape(reshape(im3,[],48)*w,size(gts_pb{id})))
subplot(222),imagesc(gts_pb{id})
subplot(223),imagesc(pb{id})

U_occ(reshape(reshape(im3,[],48)*w,size(gts_pb{id})),gts{id})
tmp=pb{id};tmp(tmp<0.5)=0;U_occ(tmp,gts{id})