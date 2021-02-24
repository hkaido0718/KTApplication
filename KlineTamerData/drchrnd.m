function draw = drchrnd(p)
	% generic code for drawing from Dirichlet distribution
	gam = gamrnd(p,1);
	draw = gam ./ sum(gam);