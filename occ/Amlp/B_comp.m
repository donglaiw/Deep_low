% initial error
psz = 17;
psz2 = 9;

psz_c = (1+psz^2)/2;
psz2_c = (1+psz2^2)/2;

psz_d = (psz-psz2)/2;
ind_mat = reshape(1:psz^2,[psz psz]);
ind_mat = ind_mat(psz_d+1:end-psz_d,psz_d+1:end-psz_d);
ind_mat = ind_mat(:);

tid =1;

for did=0:1
	fid = fopen(['mlp_st_' num2str(did) 'sx.bin'], 'rb');
	mat_x = fread(fid, [289 inf], 'single');
	fclose(fid);
	fid = fopen(['mlp_st_' num2str(did) 'sy.bin'], 'rb');
	mat_y = fread(fid, [81 inf], 'single');
	fclose(fid);
	switch tid
	case 1
		% initial error
		mean((mat_x(psz_c,:) - mat_y(psz2_c,:)).^2)
		% 0.0349,0.0357
	case 2
		% L2 error
		if did==0
			w= mat_x(ind_mat,:)'\(mat_y(psz2_c,:)');
			save('ww','w')
			mean((w'*mat_x(ind_mat,:)-mat_y(psz2_c,:)).^2)
			else
			load('ww','w')
			mean((w'*mat_x(ind_mat,:)-mat_y(psz2_c,:)).^2)
			% 0.0055,0.0055
		end		
	end
end