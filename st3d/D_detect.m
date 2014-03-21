
mex -g /home/Stephen/Desktop/VisionLib/Donglai/Mid/Boundary/SketchTokens/stDetectMex.cpp
load d_detect
    S = stDetectMex( chns, chnsSs, model.thrs, model.fids, model.child, ...
      model.distr, cids1, cids2, stride, opts.radius, nChnFtrs );
%{

max(cids1),max(cids2)
prod(sz(1:3))*2
%}

