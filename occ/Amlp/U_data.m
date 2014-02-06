% st:
%{
addpath(genpath('/home/Stephen/Desktop/VisionLib/Donglai/Mid/Boundary/SketchTokens'))
load('models/forest/modelFull.mat');
addpath('/home/Stephen/Desktop/Data/Seg/BSR')
for did=0:2
	load(['dn_ucb' num2str(did)],'Is')	
	pb = cell(1,numel(Is));
	parfor i=1:numel(Is); st = stDetect( Is{i}, model ); pb{i} = stToEdges( st, 1 );end
		save(['bd_st' num2str(did)],'pb')
end

for did = 0:2
load(['dn_ucb' num2str(did)],'gts')
gts_pb = cell(1,numel(gts));
parfor i=1:numel(gts)
    gts_pb{i} = mean(single(cat(3,gts{i}{:})),3);
end
save(['dn_ucb_pb' num2str(did)],'gts_pb')
end

end

%}
stream=RandStream('mrg32k3a','Seed',1);
set(stream,'Substream',1);
RandStream.setGlobalStream( stream );

for did = 0:1
load(['bd_st' num2str(did)])
load(['dn_ucb_pb' num2str(did)])
psz = 17;
psz2 = 9;
psz_d = (psz-psz2)/2
ind_mat = reshape(1:psz^2,[psz psz]);
ind_mat = ind_mat(psz_d+1:end-psz_d,psz_d+1:end-psz_d);
ind_mat = ind_mat(:);

mat_x = cell(1,numel(gts_pb));
mat_y = cell(1,numel(gts_pb));
num_perimg = 10000;
parfor i=1:numel(gts_pb)
	
	mat_x{i} = im2col(pb{i},[psz psz]);
	mat_y{i} = im2col(gts_pb{i},[psz psz]);
	
	tmp_ind = randsample(size(mat_x{i},2),num_perimg);
	mat_y{i} = mat_y{i}(ind_mat,tmp_ind);
	mat_x{i} = mat_x{i}(:,tmp_ind);
end
mat_x = cell2mat(mat_x);
mat_y = cell2mat(mat_y);
%save(['mlp_st' num2str(did)],'-v7.3','mat_x','mat_y')

fid = fopen(['mlp_st_' num2str(did) 'x.bin'], 'wb');
fwrite(fid, mat_x, 'single');
fclose(fid);
fid = fopen(['mlp_st_' num2str(did) 'y.bin'], 'wb');
fwrite(fid, mat_y, 'single');
fclose(fid);
%{

fid = fopen(['mlp_st_' num2str(did) 'x.bin'], 'rb');
mat_x = fread(fid, [289 inf], 'single');
fclose(fid);
fid = fopen(['mlp_st_' num2str(did) 'y.bin'], 'rb');
mat_y = fread(fid, [81 inf], 'single');
fclose(fid);


fid = fopen(['mlp_st_' num2str(did) 'sx.bin'], 'wb');
fwrite(fid, mat_x(:,1:10:end), 'single');
fclose(fid);
fid = fopen(['mlp_st_' num2str(did) 'sy.bin'], 'wb');
fwrite(fid, mat_y(:,1:10:end), 'single');
fclose(fid);

	%}
end




