
%DATA_VOC = '/data/vision/billf/manifold-learning/DL/Data/VOC2012/JPEGImages/';
DATA_VOC = '/home/Stephen/Desktop/Data/Classification/VOC/VOC2012/JPEGImages/';
imgIds=dir([DATA_VOC '*.jpg']);
imgIds={imgIds.name};

% extract commonly used options
psz=17;
id=1e4+10;
I = rgb2gray(imread([DATA_VOC imgIds{id}]));
sz = size(I);
ps = im2col(I,[psz psz]);
nps = im2col(uint8(single(I)+single(rand(sz)*sig)),[psz psz]);
sig = 25;
save(['data/test/test_' num2str(id)],'ps','nps');
done = 0
if done
    load dl_r0_1_10
    c=scol2im(result{1},psz,sz(1),sz(2),'average');
    imwrite(uint8(255*c),'ha.png')
    load(['data/test/test_' num2str(id)],'ps','nps');
    imwrite(uint8(I),'hac.png')
    imwrite(uint8(nps),'han.png')
end
