

% de-mean
mm = mean(single(mat_x),2);
mat_x = bsxfun(@minus,single(mat_x),mm);
save ../data/train/ucb_st_15_dc.mat mat_x mat_y

% four direction
psz = 15;
ind0 = reshape(1:psz^2*3,[psz psz 3]);
ind1=ind0;ind2=ind0;ind3=ind0;
for i=1:3
    ind1(:,:,i) = rot90(ind0(:,:,i));
    ind2(:,:,i) = rot90(ind1(:,:,i));
    ind3(:,:,i) = rot90(ind2(:,:,i));
end
mat_x = [mat_x mat_x(ind1(:),:) mat_x(ind2(:),:) mat_x(ind3(:),:)];
mat_y = [mat_y mat_y mat_y mat_y];
save ../data/train/ucb_st_15_4d.mat mat_x mat_y
