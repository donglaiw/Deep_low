%load /home/Stephen/Desktop/Data/Seg/BSR/dn_ucb_pb0
%load ../bd_st0


psz = 31;
algo_id = 1;
ssz = 0;
for id=1:200;
id
for gid = 0:1;
switch gid
case 0
    sub_im = pb{id}((ssz+1):end-ssz,(ssz+1):end-ssz);
case 1
    sub_im = gts_pb{id}((ssz+1):end-ssz,(ssz+1):end-ssz);
end
mat_x2 = im2col(sub_im,[psz psz]);

for Did = 0:1;
switch Did
case 0
    load sc_D
case 1
    load km_D
end
switch algo_id
case 1 
    param.L= 5; % not more than 10 non-zeros coefficients
    param.eps=0.001; % squared norm of the residual should be less than 0.1
    a2=mexOMP(double(mat_x2),D,param);
case 2
    param.lambda=0.15; % not more than 20 non-zeros coefficients
    param.mode=1;        % penalized formulation
    a2=mexLasso(double(mat_x2),D,param);
end
im2 = scol2im(D*a2,psz,size(sub_im,1),size(sub_im,2),'average');
%subplot(211),imagesc(sub_im)
%subplot(212),imagesc(im2)
imwrite(sub_im,sprintf('in_%d_%d_%d.png',id,Did,gid))
imwrite(im2/max(im2(:)),sprintf('out_%d_%d_%d.png',id,Did,gid))
save(sprintf('%d_%d_%d',id,Did,gid),'im2','sub_im')

end
end
end
