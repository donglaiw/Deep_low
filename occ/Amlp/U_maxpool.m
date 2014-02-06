function y=U_maxpool(im,ksz,ksd)


sz = size(im);
ind = reshape(1:numel(im),sz);
id = ind(1:ksd:end-ksz+1,1:ksd:end-ksz+1);

p_ind = bsxfun(@plus, (0:ksz-1), (0:ksz-1)*sz(1));

y=reshape(max(im(bsxfun(@plus,id(:),p_ind(:)')),[],2),size(id));
