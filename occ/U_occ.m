function [y,cc] = U_occ(pb,gt_bd,thresh)
if ~exist('thresh','var')
thresh = 0:0.02:1;
end
sz = size(pb);
sz2 = size(gt_bd);
if sum(sz~=sz2)>0
    pb = imresize(pb,sz2,'nearest');
end
nthres = numel(thresh);
sumR = zeros(1,nthres);
    cntR = zeros(1,nthres);
    sumP = zeros(1,nthres);
    cntP = zeros(1,nthres);
    for t = 1:numel(thresh)
        % threshold pb to get binary boundary map
        bmap = (pb>=thresh(t));
        % thin the thresholded pb to make sure boundaries are standard thickness
        bmap = double(bwmorph(bmap,'thin',inf));
        % accumulate machine matches, since the machine pixels are
        % allowed to match with any segmentation
        accP = zeros(size(pb));
        % compare to each seg in turn
        % compute the correspondence
        [match1,match2] = correspondPixels(bmap,gt_bd);
        % accumulate machine matches
        accP = accP | match1;
        % compute recall
        sumR(t) = sum(gt_bd(:));
        cntR(t) = sum(match2(:)>0);
        % compute precision
        sumP(t) = sumP(t) + sum(bmap(:));
        cntP(t) = cntP(t) + sum(accP(:));
    end

r = cntR./(sumR+(sumR==0));
p = cntP./(sumP+(sumP==0));
y = U_roc(r,p);
cc = [sumR;cntR;sumP;cntP]';
