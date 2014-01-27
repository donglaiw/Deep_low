%x=150;y=120;
x=170;y=150;

p_sz=35;
p_szh=floor((p_sz-1)/2);
im_id =1;
sz = size(Is{im_id});

p_id = bsxfun(@plus,(-p_szh:p_szh)',sz(1)*(-p_szh:p_szh));
im = pb0;
im = Is{im_id};
db0 =im(sub2ind(sz,x,y)+p_id);
db = [0; reshape(db0',[],1)];
%db = [0; db0(:)];
imagesc(db0)
pb0(x,y)
save data/test/db db 
