% Replication files for "Bayesian inference in a class of partially identified models" by Brendan Kline and Elie Tamer
% Simulation for game

clear variables
close all

computeTrueIDset = 0

diary(strcat('output/simulation_game', int2str(computeTrueIDset), '_diary.txt'))

computeTrueIDset

RandStream.setGlobalStream(RandStream('mt19937ar','Seed', 1));

% model parameters
delta1 = -0.5; 
delta2 = -0.5;
beta1 = 0.2;
beta2 = 0.2;
mu_eps = [0 0];
sigma_eps = [1 0.5; 0.5 1];

if computeTrueIDset,
	total_reps = 1;
	color_list = jet(total_reps);
	numdraws = 1;
	numdraws_extra = 1;
	slicedraw_num = 7500;
	max_figs = 6;
else,
	total_reps = 250;
	color_list = jet(total_reps);
	n = 500;
	numdraws = 100;
	numdraws_extra = numdraws + 1;
	slicedraw_num = 7500;
	max_figs = 6;
end;

credset = cell(total_reps,4);
credset{total_reps,4} = [];
credset_cover{1}(1:total_reps) = 0;
credset_cover{2}(1:total_reps) = 0;
credset_cover{3}(1:total_reps) = 0;
credset_cover{4}(1:total_reps) = 0;
cred_level = 0.95;

% setup parallel pool for 4 workers 
parpoollocal = parpool(4);
spmd
	workerstream = RandStream('mlfg6331_64','Seed',1);
	RandStream.setGlobalStream(workerstream);
end

tic;
for reps = 1:total_reps,
	% generate the data
		if computeTrueIDset,
			x = [beta1 beta2 delta1 delta2 sigma_eps(1,2)];
			sigma = [1 x(5); x(5) 1];

			p11 = qsimvnv(50000, sigma, [-x(1)-x(3)  -x(2)-x(4)], [Inf Inf]);
			p00 = qsimvnv(50000, sigma, [-Inf -Inf], [-x(1)  -x(2)]);

			model_01_uniq = qsimvnv(50000, sigma, [-Inf -x(2)], [-x(1)  Inf]) + qsimvnv(50000, sigma, [-x(1)   -x(2)-x(4)], [-x(1)-x(3)  Inf]);
			model_multiple = qsimvnv(50000, sigma, [-x(1) -x(2)], [-x(1)-x(3)  -x(2)-x(4)]);
			s = 0.5;
			p01 = model_01_uniq+s*model_multiple;

			p10 = 1 - p11 - p00 - p01;
			
		else,
			eps = mvnrnd(mu_eps,sigma_eps,n);

			for i = 1:n
			    if beta1+delta1+eps(i,1)>=0 && beta2+delta2+eps(i,2)>=0
				y(i,:) = [1 1];

			    elseif beta1+eps(i,1)<=0 && beta2+eps(i,2)<=0
				y(i,:) = [0 0];

			    elseif ((beta1+eps(i,1)<0 && beta2+eps(i,2)>0)) || (-beta1<=eps(i,1) && eps(i,1)<=-beta1-delta1 && eps(i,2)>=-beta2-delta2)
				y(i,:) = [0 1];

			    elseif (beta1+eps(i,1)>0 && beta2+eps(i,2)<0) || (-beta2<=eps(i,2) && eps(i,2)<=-beta2-delta2 && eps(i,1)>=-beta1-delta1)
				y(i,:) = [1 0];

			    else
				r = binornd(1, 0.5);
				if r
				    y(i,:) = [0 1];
				else
				    y(i,:) = [1 0];
				end;
			    end;
			end;

			p11=0; p10=0; p00=0; p01=0;

			for i = 1:n
			   if y(i,:) == [1 1]
			       p11 = p11+1;
			   elseif y(i,:) == [1 0]
			       p10 = p10+1;
			   elseif y(i,:) == [0 1]
			       p01 = p01+1;
			   elseif y(i,:) == [0 0]
			       p00 = p00+1;
			   end;
			end;
			
			p11 = p11/n;
			p10 = p10/n;
			p01 = p01/n;
			p00 = p00/n;
		end;
	
	% draws from the posterior
	good_draws_storage(1:numdraws) = 0;
	x0 = [0.3 0.3 -0.3 -0.3 0.3];
	slicedraws_raw = cell(numdraws_extra,1);
	slicedraws_raw{numdraws_extra} = [];
	slicedraws = cell(numdraws_extra,1);
	slicedraws{numdraws_extra} = [];
	minbeta1(1:numdraws_extra) = NaN;
	maxbeta1(1:numdraws_extra) = NaN;
	minbeta2(1:numdraws_extra) = NaN;
	maxbeta2(1:numdraws_extra) = NaN;
	mindelta1(1:numdraws_extra) = NaN;
	maxdelta1(1:numdraws_extra) = NaN;
	mindelta2(1:numdraws_extra) = NaN;
	maxdelta2(1:numdraws_extra) = NaN;
	minrho(1:numdraws_extra) = NaN;
	maxrho(1:numdraws_extra) = NaN;
				
	parfor draws = 1:numdraws_extra;
	
		workerstream = RandStream.getGlobalStream;
		workerstream.Substream = 1 + 10000*reps + 100000*(draws - 1);
		
		if computeTrueIDset,
			mu = [p11 p10 p01 p00];
			tol = 0.0015;
			tol2 = 0.0015;
		else,
			if draws <= numdraws,
				mu = drchrnd([p11 p10 p01 p00]*n+[1 1 1 1]);
				tol = 0.0015;
				tol2 = 0.0015;
			else,
				mu = [p11 p10 p01 p00];
				tol = 0.0015;
				tol2 = 0.0015;
			end;
		end;

		% objective function.... parameter is x --- [beta1 beta2 delta1 delta2 rho]
			T = 0.1;
			obj_fn = @(x)criterion_fn([x mu], 10000, 1);
			dens_T = @(x)exp(-obj_fn(x)/T);
			neg_dens_T = @(x) -dens_T(x);
			dens_indic = @(x)1*(criterion_fn([x mu], tol, 1) <= tol);
			dens_indic2 = @(x)1*(criterion_fn([x mu], tol2, 1) <= tol2);
			
			% find point in identified set using the "smoothed" version of the indicator for the criterion function
			x0d = x0;
			eval_at_x0d = dens_indic(x0d);
			total_min_tries = 1;
			width_x0 = 0.05;
			opt = optimset('Display','none','TolX', 10e-10, 'LargeScale', 'off', 'MaxFunEvals', 2500);
			
			while eval_at_x0d == 0 && total_min_tries <= 2500,
				rand_moves = rand(1,size(x0,2)-1);
				rand_moves = [rand_moves(1:3) rand_moves(3) rand_moves(4)];
				delta_x0 = width_x0*rand_moves-width_x0/2;
				try
					x0d = fminunc(neg_dens_T, min(max(x0d + delta_x0,[-1.5 -1.5 -1.5 -1.5 0]),[1.5 1.5 0 0 1]), opt);
				catch err
					disp(err.message);
 					disp(err.identifier);
					min(max(x0d + delta_x0,[-1.5 -1.5 -1.5 -1.5 0]),[1.5 1.5 0 0 1])
					neg_dens_T(min(max(x0d + delta_x0,[-1.5 -1.5 -1.5 -1.5 0]),[1.5 1.5 0 0 1]))
					fprintf(1, '\n Bad attempt to minimize: Rep: %d Draw: %d \n', reps, draws);
				end
				eval_at_x0d = dens_indic(x0d);
				total_min_tries = total_min_tries + 1;
			end;

		try
			% slice sampling using the indicator for the identified set based on the criterion function
			slicedraws_raw{draws} = slicesample(x0d,slicedraw_num,'pdf',dens_indic,'burnin', 10, 'width', 0.25*[1 1 1 1 1]);
			x0d
		catch err
			if (strcmp(err.identifier,'stats:slicesample:BadInitial'))
				fprintf(1, '\n Bad initial for slice sampler: Rep: %d Draw: %d \n', reps, draws);
				[draws x0d]
				slicedraws_raw{draws}(1:slicedraw_num, 1:size(x0,2)) = -1000; 
			end
		end

		if tol2 >= tol && slicedraws_raw{draws}(1, 1) ~= -1000,
			index = true(1,slicedraw_num);
		else,
			index = false(1, slicedraw_num);
			for slicedraw_check = 1:slicedraw_num,
				if dens_indic2(slicedraws_raw{draws}(slicedraw_check,:)),
					index(slicedraw_check) = true;
				end;
			end;
		end;

		if max(index) == 1,
			slicedraws{draws} = slicedraws_raw{draws}(index,:);
			
			if draws <= numdraws,
				good_draws_storage(draws) = 1;
			end;

			minbeta1(draws) = min(slicedraws{draws}(:,1));
			maxbeta1(draws) = max(slicedraws{draws}(:,1));
			minbeta2(draws) = min(slicedraws{draws}(:,2));
			maxbeta2(draws) = max(slicedraws{draws}(:,2));

			mindelta1(draws) = min(slicedraws{draws}(:,3));
			maxdelta1(draws) = max(slicedraws{draws}(:,3));
			mindelta2(draws) = min(slicedraws{draws}(:,4));
			maxdelta2(draws) = max(slicedraws{draws}(:,4));

			minrho(draws) = min(slicedraws{draws}(:,5));
			maxrho(draws) = max(slicedraws{draws}(:,5));
		else,
			slicedraws{draws} = [-Inf -Inf -Inf -Inf -Inf];	

			minbeta1(draws) = Inf;
			maxbeta1(draws) = -Inf;
			minbeta2(draws) = Inf;
			maxbeta2(draws) = -Inf;

			mindelta1(draws) = Inf;
			maxdelta1(draws) = -Inf;
			mindelta2(draws) = Inf;
			maxdelta2(draws) = -Inf;

			minrho(draws) = Inf;
			maxrho(draws) = -Inf;
		end;

		fprintf(1, '%2.0f ... ', draws);
		if mod(draws,10) == 0,
			fprintf(1, '\n');
		end;
	end;
	
	reps
	% compute credible set (meaning a set that contains the identified set with specified probability)
	% note that if empty identified set at given mu, then count as length 0, and [Inf -Inf] identified set
		for parm = 1:4,
			if parm==1,
				mincred_start = minbeta1(numdraws_extra);
				maxcred_start = maxbeta1(numdraws_extra);
				mincred_bnd = minbeta1(1:numdraws);
				maxcred_bnd = maxbeta1(1:numdraws);
				truelower = 0;
				trueupper = 0.75;
			elseif parm==2,
				mincred_start = minbeta2(numdraws_extra);
				maxcred_start = maxbeta2(numdraws_extra);
				mincred_bnd = minbeta2(1:numdraws);
				maxcred_bnd = maxbeta2(1:numdraws);	
				truelower = 0;
				trueupper = 0.75;			
			elseif parm==3,
				mincred_start = mindelta1(numdraws_extra);
				maxcred_start = maxdelta1(numdraws_extra);
				mincred_bnd = mindelta1(1:numdraws);
				maxcred_bnd = maxdelta1(1:numdraws);
				truelower = -1.50;
				trueupper = -0.04;				
			elseif parm==4,
				mincred_start = mindelta2(numdraws_extra);
				maxcred_start = maxdelta2(numdraws_extra);
				mincred_bnd = mindelta2(1:numdraws);
				maxcred_bnd = maxdelta2(1:numdraws);
				truelower = -1.50;
				trueupper = -0.04;		
			end;
				
			for bound = 0:1000,
				
				credset{reps,parm}(1) = mincred_start - bound/1000;
				credset{reps,parm}(2) = maxcred_start + bound/1000;

				credset_bayescover_comp(1:numdraws) = 0;
				for draws = 1:numdraws,
					if credset{reps,parm}(1) <= mincred_bnd(draws) && credset{reps,parm}(2) >= maxcred_bnd(draws),
						credset_bayescover_comp(draws) = 1;
					end;
				end;
				if (mean(credset_bayescover_comp) >= cred_level-0.005 && mean(credset_bayescover_comp) <= cred_level+0.005) || mean(credset_bayescover_comp) > cred_level+0.005,
					break;
				end;
			end;
			
			fprintf(1, 'Parm: %2.0f Bayes credibility: %3.3f ... Bayes cred set: %3.3f %3.3f ... Starting point: %3.3f %3.3f \n', parm, mean(credset_bayescover_comp), credset{reps,parm}, mincred_start, maxcred_start);
			if credset{reps,parm}(1) <= truelower && trueupper <= credset{reps,parm}(2),
				credset_cover{parm}(reps) = 1;
			end;
			fprintf(1, 'Frequentist coverage: %3.3f \n', mean(credset_cover{parm}(1:reps)));
		end;

	numgridpts = 500;
	grid = sort([linspace(-4.5,3.5,numgridpts) 0 1]);
	inidents(1:size(grid,2)) = 0;
	good_draws_indices = find(good_draws_storage == 1);

	variables_for_plots = {'beta1', 'beta2', 'delta1', 'delta2', 'rho'}

	% the following plots the marginal "posterior curves" across the draws from the DGP
	if mod(reps,10) == 0,
		for var_i = 1:5,
			figure(var_i);
			varname = [variables_for_plots{var_i}];
			inidents(1:size(grid,2)) = 0;
			for draws = good_draws_indices,
				eval(['lowerbnd = min' varname '(draws);']);
				eval(['upperbnd = max' varname '(draws);']);
				gridlb = max(find(grid <= lowerbnd));
				gridub = min(find(grid >= upperbnd));
				inidents(gridlb:gridub) = inidents(gridlb:gridub) + 1;
			end;
			inidents = inidents/numdraws;

			if var_i == 1 || var_i == 2,
				axis([-1.5 2 0 1.1]);
			elseif var_i == 3 || var_i == 4,
				axis([-3 0 0 1.1]);
			elseif var_i == 5,
				axis([0 1 0 1.1]);
			end;

			hold on;
			
			% posterior probability of theta in the identified set
			plot(grid, inidents, 'color',color_list(reps,:));
			hold on;
			if var_i <= 4,
				plot(credset{reps,var_i}(1), 0, 'o','color',color_list(reps,:));
				hold on;
				plot(credset{reps,var_i}(2), 0, 'o','color',color_list(reps,:));
				hold on;
			end;
		end;
	end;
	
	% joint posterior contour for just one draw from the DGP (first draw)
	if reps == 1,
		centers = {-1:0.1:2 -2:0.1:0};
		P = cell(length(good_draws_indices),1);
		P{length(good_draws_indices),1} = [];
		for draws = good_draws_indices,
			P{draws}=histproj_fixedgrid([slicedraws{draws}(:,1) slicedraws{draws}(:,3)],centers,0.02);
		end;
		Psum = P{good_draws_indices(1)};
		for draws = good_draws_indices(2:length(good_draws_indices)),
			Psum = Psum + P{draws};
		end;
		Psum = Psum/numdraws;
		figure(6);
		
		if computeTrueIDset,
			contour(centers{1}, centers{2}, Psum, 1, 'LineColor', [0 0 0]);
		else,
			C = contour(centers{1}, centers{2}, Psum);
			colorbar
		end;
		xlabel('\beta_{1}');
		ylabel('\Delta_{1}');
	end;
	
	curtime = toc;
	fprintf(1, 'Time left: %3.2f minutes ... \n', curtime/reps*(total_reps-reps)/60);

end;

delete(parpoollocal);

save(strcat('output/simulationgamefigs_workspace',int2str(computeTrueIDset),'.mat'))

for var_i = 1:max_figs,
	figure(var_i);
	set(gca, 'LooseInset', get(gca, 'TightInset'));
	set(gcf, 'PaperPosition', [0 0 5 5]);
	set(gcf, 'PaperSize', [5 5]);
	print('-dpdf',strcat('output/game', int2str(computeTrueIDset), int2str(var_i), '.pdf'))
end;

zip('output/simulationgamefigs', {'output/game*.pdf'});

diary off