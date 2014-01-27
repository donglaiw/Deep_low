function [im,bbox]=U_bb(im,opt)
sz = size(im);
bbox =[];
switch opt.id
case 1
    % keep ratio
    rr = opt.CROP_DIM./sz(1:2);
    im = imresize(squeeze(im),max(rr));
    sz=size(im);
    off = floor((sz-opt.CROP_DIM)/2);
    im = uint8(im(off(1)+(1:opt.CROP_DIM),off(2)+(1:opt.CROP_DIM),:));
case 2
    % foreground only
    [aa,bb] = bwlabel(opt.seg==opt.cl_id);
    cc = histc(aa(:),1:bb);
    [~,dd] = max(cc);
    [xx,yy] = ind2sub(sz,find(aa==dd));
    bbox = [min(xx(:)) max(xx(:)) min(yy(:)) max(yy(:))];
    bbox([1 3]) = max(1,bbox([1 3]) - opt.BD_DIM);
    bbox(2) = min(sz(1),bbox(2) + opt.BD_DIM);
    bbox(4) = min(sz(2),bbox(4) + opt.BD_DIM);
    opt.id=1;
    im = U_bb(im(bbox(1):bbox(2),bbox(3):bbox(4),:),opt);
case 3
    opt.id = 1;
    % propose four images
    sz2 = floor(opt.Corner*sz(1:2));
    ims = zeros([sz2 3 4]);
    segs = zeros([sz2 4]);
    ims(:,:,:,1) = im(1:sz2(1),1:sz2(2),:);
    ims(:,:,:,2) = im(1:sz2(1),(1+end-sz2(2)):end,:);
    ims(:,:,:,3) = im((end-sz2(1)+1):end,1:sz2(2),:);
    ims(:,:,:,4) = im((end-sz2(1)+1):end,(1+end-sz2(2)):end,:);
    segs(:,:,1) = opt.seg(1:sz2(1),1:sz2(2));
    segs(:,:,2) = opt.seg(1:sz2(1),(1+end-sz2(2)):end);
    segs(:,:,3) = opt.seg((end-sz2(1)+1):end,1:sz2(2));
    segs(:,:,4) = opt.seg((end-sz2(1)+1):end,(1+end-sz2(2)):end);
    cc = sum(sum(segs==0|segs==255,1),2); 
    [~,sid] = max(cc);
    % remove some row,col
    im=U_bb(ims(:,:,:,sid),opt);
end

end
