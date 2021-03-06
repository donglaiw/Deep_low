addpath('/home/Stephen/Desktop/VisionLib/Donglai/Util/io')



did =1;
% 0. data structure
switch did
    case 0
        DD='/home/Stephen/Desktop/Data/Occ/Berk/train/';
        fns = dir(DD);
        fns(1:2)=[];
        gts = cell(1,numel(fns));
        gts2 = cell(1,numel(fns));
        ids = zeros(1,numel(fns));
        for i=1:numel(fns)
            tmp_n=dir([DD fns(i).name '/*.mat']);
            tmp_n=tmp_n(1).name;
            ids(i) = str2num(tmp_n(find(tmp_n=='e',1,'first')+1:find(tmp_n=='.',1,'first')-1));
            tmp=load([DD fns(i).name '/' tmp_n]);
            gts{i}=tmp.groundTruth.Boundaries;
            gts2 = cell(1,numel(fns));
        end
        
    case 1
        % for i in `ls *.bmp`;do convert $i "${i%.bmp}.png"; done;
        
        DD='/home/Stephen/Desktop/Data/Seg/Segtrack/';
        fns = dir(DD);
        fns(1:2)=[];
        
        
        gts = cell(1,numel(fns));
        gts2 = cell(1,numel(fns));
        for i=1:numel(fns)
            tmp_fn = U_getims([DD fns(i).name '/ground-truth/'],'png');
            im = imread([DD fns(i).name '/ground-truth/' tmp_fn(1).name]);
            sz = size(im);
            tmp_e = zeros([sz(1:2) numel(tmp_fn)]);
            tmp_e2  = zeros([sz(1:2) numel(tmp_fn)]);
            parfor j= 1:numel(tmp_fn)
                im = imread([DD fns(i).name '/ground-truth/' tmp_fn(j).name]);
                [gx,gy]=gradient(double(im(:,:,1)));
                tmp_e(:,:,j) = (gx~=0)|(gy~=0);
                tmp_e2(:,:,j) = bwdist(tmp_e(:,:,j)>0)<2;
            end
            gts{i} = logical(tmp_e);
            gts2{i} = logical(tmp_e2);
        end
end
save segtrack gts gts2



% for i in `ls *.bmp`;do convert $i "${i%.bmp}.png"; done;
% for i in `ls *.png`;do convert $i "${i%.png}.bmp"; done;
% 1. kmeans


save -v7.3 segtrack_mat mat_x mat_x2 mat_y
