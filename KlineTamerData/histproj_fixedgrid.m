function P = histproj_fixedgrid(x,g,lw)

	H = hist3(x, g);

	maxdens = max(max(H));
	lower = lw*maxdens;

	Hcont = zeros(size(g{1},2), size(g{2},2));

	for i = 1:size(Hcont,1),
	    for j = 1:size(Hcont,2),
			if H(i,j) >= lower,
				Hcont(i,j) = 1;
			end;
	    end;
	end;

	P=Hcont';
	
end

