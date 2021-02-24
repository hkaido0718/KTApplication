% Replication files for "Bayesian inference in a class of partially identified models" by Brendan Kline and Elie Tamer
% Empirical application to entry in airline markets

diary('output/empirical_game_diary.txt')

clear variables
close all

tic;

% load data
M = csvread('airlinedata.dat');
XX(:,1) = M(:,1);
XX(:,2) = M(:,2);
XX(:,3) = M(:,3);
y(:,1) = M(:,4);
y(:,2) = M(:,5);
n = size(y,1);

% parameters for posterior simulation and model specification
min_model_spec = 1;
max_model_spec = 4;
numdraws = 250;
numdraws_extra = numdraws + 1;
slicedraw_num = 5000;
cred_level = 0.95;

% computed regressors
	LCCmedianPres = median(XX(:,1));
	OAmedianPres = median(XX(:,2));
	XX(:,4) = (XX(:,1) >= LCCmedianPres);
	XX(:,5) = (XX(:,2) >= OAmedianPres);
	
	sizeMedian = median(XX(:,3));
	XX(:,6) = (XX(:,3) >= sizeMedian);

for model_spec = [1 2 4],

	clearvars -except M XX y n numdraws numdraws_extra slicedraw_num model_spec min_model_spec max_model_spec cred_level

	stream = RandStream('mlfg6331_64','Seed',1);
	RandStream.setGlobalStream(stream);

	if model_spec == 1,
		p11=0; p10=0; p00=0; p01=0;
		n_total = n;

		for i=1:n
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
		p11 = p11/n_total
		p10 = p10/n_total
		p01 = p01/n_total
		p00 = p00/n_total
	elseif model_spec == 2,		
		p11(1:2,1:2) = 0; p10(1:2,1:2)=0; p00(1:2,1:2)=0; p01(1:2,1:2)=0;
		n_total(1:2,1:2) = 0;
		for LCCpres = 0:1,
			for OApres = 0:1,
				for i = 1:n,
					if XX(i,4) == LCCpres && XX(i,5) == OApres,
						if y(i,:) == [1 1]
							p11(LCCpres+1,OApres+1) = p11(LCCpres+1,OApres+1)+1;
						elseif y(i,:) == [1 0]
							p10(LCCpres+1,OApres+1) = p10(LCCpres+1,OApres+1)+1;
						elseif y(i,:) == [0 1]
							p01(LCCpres+1,OApres+1) = p01(LCCpres+1,OApres+1)+1;
						elseif y(i,:) == [0 0]
							p00(LCCpres+1,OApres+1) = p00(LCCpres+1,OApres+1)+1;
						end;
						n_total(LCCpres+1,OApres+1) = n_total(LCCpres+1,OApres+1)+1;
					end;
				end;
				p11(LCCpres+1,OApres+1) = p11(LCCpres+1,OApres+1)/n_total(LCCpres+1,OApres+1);
				p10(LCCpres+1,OApres+1) = p10(LCCpres+1,OApres+1)/n_total(LCCpres+1,OApres+1);
				p01(LCCpres+1,OApres+1) = p01(LCCpres+1,OApres+1)/n_total(LCCpres+1,OApres+1);
				p00(LCCpres+1,OApres+1) = p00(LCCpres+1,OApres+1)/n_total(LCCpres+1,OApres+1);
			end;
		end;
	elseif model_spec == 4,
		for LCCpres = 0:1,
			for OApres = 0:1,
				for sizei = 0:1,
					n_total{LCCpres+1,OApres+1,sizei+1} = 0;
					p11{LCCpres+1,OApres+1,sizei+1} = 0; p10{LCCpres+1,OApres+1,sizei+1}=0; p00{LCCpres+1,OApres+1,sizei+1}=0; p01{LCCpres+1,OApres+1,sizei+1}=0;
					for i = 1:n,
						if XX(i,4) == LCCpres && XX(i,5) == OApres && XX(i,6) == sizei,
							if y(i,:) == [1 1]
								p11{LCCpres+1,OApres+1,sizei+1} = p11{LCCpres+1,OApres+1,sizei+1}+1;
							elseif y(i,:) == [1 0]
								p10{LCCpres+1,OApres+1,sizei+1} = p10{LCCpres+1,OApres+1,sizei+1}+1;
							elseif y(i,:) == [0 1]
								p01{LCCpres+1,OApres+1,sizei+1} = p01{LCCpres+1,OApres+1,sizei+1}+1;
							elseif y(i,:) == [0 0]
								p00{LCCpres+1,OApres+1,sizei+1} = p00{LCCpres+1,OApres+1,sizei+1}+1;
							end;
							n_total{LCCpres+1,OApres+1,sizei+1}=n_total{LCCpres+1,OApres+1,sizei+1}+1;
						end;
					end;
					p11{LCCpres+1,OApres+1,sizei+1} = p11{LCCpres+1,OApres+1,sizei+1}/n_total{LCCpres+1,OApres+1,sizei+1};
					p10{LCCpres+1,OApres+1,sizei+1} = p10{LCCpres+1,OApres+1,sizei+1}/n_total{LCCpres+1,OApres+1,sizei+1};
					p01{LCCpres+1,OApres+1,sizei+1} = p01{LCCpres+1,OApres+1,sizei+1}/n_total{LCCpres+1,OApres+1,sizei+1};
					p00{LCCpres+1,OApres+1,sizei+1} = p00{LCCpres+1,OApres+1,sizei+1}/n_total{LCCpres+1,OApres+1,sizei+1};
				end;
			end;
		end;	
	
	end;

	if model_spec == 1,
		beta1 = 0.2;
		beta2 = 0.2;
		delta1 = -1.5;
		delta2 = -1.5;
		rho = 0.8;
		x0 = [beta1 beta2 delta1 delta2 rho];

	elseif model_spec == 2,
		beta1cons = 0.5;
		beta2cons = 1.1;
		beta1pres = 1.3;
		beta2pres = 0.2;
		delta1 = -2;
		delta2 = -1.4;
		rho = 0.5;
		x0 = [beta1cons beta2cons beta1pres beta2pres delta1 delta2 rho];

	elseif model_spec == 4,
		beta1cons = -1;
		beta2cons = 0.5;
		beta1size = 0.4;
		beta2size = 0.4;
		beta1pres = 1.5;
		beta2pres = 0.5;
		delta1 = -1.2;
		delta2 = -1.2;
		rho = 0.8;
		x0 = [beta1cons beta2cons beta1size beta2size beta1pres beta2pres delta1 delta2 rho];		
	end;
	
	slicedraws_raw = cell(numdraws_extra,1);
	slicedraws_raw{numdraws_extra} = [];
	slicedraws = cell(numdraws_extra,1);
	slicedraws{numdraws_extra} = [];
	minbeta1(1:numdraws_extra) = NaN;
	maxbeta1(1:numdraws_extra) = NaN;
	minbeta2(1:numdraws_extra) = NaN;
	maxbeta2(1:numdraws_extra) = NaN;
	minbeta1cons(1:numdraws_extra) = NaN;
	maxbeta1cons(1:numdraws_extra) = NaN;
	minbeta2cons(1:numdraws_extra) = NaN;
	maxbeta2cons(1:numdraws_extra) = NaN;
	minbeta1size(1:numdraws_extra) = NaN;
	maxbeta1size(1:numdraws_extra) = NaN;
	minbeta2size(1:numdraws_extra) = NaN;
	maxbeta2size(1:numdraws_extra) = NaN;
	minbeta1pres(1:numdraws_extra) = NaN;
	maxbeta1pres(1:numdraws_extra) = NaN;
	minbeta2pres(1:numdraws_extra) = NaN;
	maxbeta2pres(1:numdraws_extra) = NaN;
	mindelta1(1:numdraws_extra) = NaN;
	maxdelta1(1:numdraws_extra) = NaN;
	mindelta2(1:numdraws_extra) = NaN;
	maxdelta2(1:numdraws_extra) = NaN;
	minrho(1:numdraws_extra) = NaN;
	maxrho(1:numdraws_extra) = NaN;	
	
	% set to correct number of workers!
	% setup parallel pool for 4 workers 
	parpoollocal = parpool(4);
	spmd
		workerstream = RandStream('mlfg6331_64','Seed',1);
		RandStream.setGlobalStream(workerstream);
	end
	
	% draws from the posterior of the identified set
	good_draws_storage(1:numdraws) = 0;
	
	parfor draws = 1:numdraws_extra,

		workerstream = RandStream.getGlobalStream;
		workerstream.Substream = 1 + 100000*(draws - 1);
		
		if model_spec == 1,
			if draws <= numdraws,
				mu = drchrnd([p11 p10 p01 p00]*n+[1 1 1 1]*1);
			else,
				mu = [p11 p10 p01 p00];
			end;
		elseif model_spec == 2,
			if draws <= numdraws,
				for LCCpres=0:1,
					for OApres=0:1,
						mu{LCCpres+1,OApres+1} = drchrnd([p11(LCCpres+1,OApres+1) p10(LCCpres+1,OApres+1) p01(LCCpres+1,OApres+1) p00(LCCpres+1,OApres+1)]*n_total(LCCpres+1,OApres+1)+[1 1 1 1]*1);
					end;
				end;
			else,
				for LCCpres=0:1,
					for OApres=0:1,
						mu{LCCpres+1,OApres+1} = [p11(LCCpres+1,OApres+1) p10(LCCpres+1,OApres+1) p01(LCCpres+1,OApres+1) p00(LCCpres+1,OApres+1)];
					end;
				end;
			end;
		elseif model_spec == 4,
			if draws <= numdraws,
				for LCCpres=0:1,
					for OApres=0:1,
						for sizei=0:1,
							mu{LCCpres+1,OApres+1,sizei+1} = drchrnd([p11{LCCpres+1,OApres+1,sizei+1} p10{LCCpres+1,OApres+1,sizei+1} p01{LCCpres+1,OApres+1,sizei+1} p00{LCCpres+1,OApres+1,sizei+1}]*n_total{LCCpres+1,OApres+1,sizei+1}+[1 1 1 1]*1);
						end;
					end;
				end;
			else,
				for LCCpres=0:1,
					for OApres=0:1,
						for sizei=0:1,
							mu{LCCpres+1,OApres+1,sizei+1} = [p11{LCCpres+1,OApres+1,sizei+1} p10{LCCpres+1,OApres+1,sizei+1} p01{LCCpres+1,OApres+1,sizei+1} p00{LCCpres+1,OApres+1,sizei+1}];
						end;
					end;
				end;
			end;
		end;

		% objective function
			if model_spec == 1,
				% parameter is x --- [beta1 beta2 delta1 delta2 rho]
				T = 0.1;
				tol = 0.01;
				tol2 = 0.01;
				lowerparmbnd = [-2.5 -2.5 -2.5 -2.5 0];
				upperparmbnd = [2.5 2.5 0 0 1];
				
			elseif model_spec == 2,
				% parameter is x --- [beta1cons beta2cons beta1pres beta2pres delta1 delta2 rho]
				T = 0.1;
				tol = 0.025;
				tol2 = 0.025;
				lowerparmbnd = [-2.5 -2.5 -2.5 -2.5 -2.5 -2.5 0];
				upperparmbnd = [2.5 2.5 2.5 2.5 0 0 1];

			elseif model_spec == 4,
				% parameter is x --- [beta1cons beta2cons beta1size beta2size beta1pres beta2pres delta1 delta2 rho]
				T = 0.5;
				tol = 0.075;
				tol2 = 0.075;
				lowerparmbnd = [-2.5 -2.5 -2.5 -2.5 -2.5 -2.5 -2.5 -2.5 0];
				upperparmbnd = [2.5 2.5 2.5 2.5 2.5 2.5 0 0 1];

			end;
	
			obj_fn = @(x)criterion_fn_modelspec(x, mu, 10000, model_spec);
			dens_T = @(x)exp(-obj_fn(x)/T);
			neg_dens_T = @(x) -dens_T(x);
			dens_indic = @(x)1*(criterion_fn_modelspec(x, mu, tol, model_spec) <= tol);
			dens_indic2 = @(x)1*(criterion_fn_modelspec(x, mu, tol2, model_spec) <= tol2);
			
		% approximate the identified set			
			
			% find point in identified set using the "smoothed" version of the indicator for the criterion function
			x0d = x0;
			eval_at_x0d = dens_indic(x0d);
			total_min_tries = 1;
			width_x0 = 0.02;
			opt = optimset('Display','none','TolX', 10e-10, 'LargeScale', 'off', 'MaxFunEvals', 2500);
			while eval_at_x0d == 0 && total_min_tries <= 500,
				rand_moves = rand(1,size(x0,2));
				delta_x0 = width_x0*rand_moves-width_x0/2;
				try
					x0d = fminunc(neg_dens_T, min(max(x0d + delta_x0,lowerparmbnd),upperparmbnd), opt);
				catch err	
					disp(err.message);
 					disp(err.identifier);
					min(max(x0d + delta_x0,lowerparmbnd),upperparmbnd)
					neg_dens_T(min(max(x0d + delta_x0,lowerparmbnd),upperparmbnd))
					fprintf(1, '\n Bad attempt to minimize: Draw: %d \n', draws);
				end
				eval_at_x0d = dens_indic(x0d);
				total_min_tries = total_min_tries + 1;
			end;
			
			try
				% slice sampling using the indicator for the identified set based on the criterion function
				slicedraws_raw{draws} = slicesample(x0d,slicedraw_num,'pdf',dens_indic,'burnin',10, 'width', ones(1,size(x0,2)));
				x0d
			catch err
				if (strcmp(err.identifier,'stats:slicesample:BadInitial'))
					fprintf(1, '\n Bad initial for slice sampler: Draw: %d Minimize attempts: %d \n', draws, total_min_tries);
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
				if model_spec == 1,
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
				elseif model_spec == 2,
					minbeta1cons(draws) = min(slicedraws{draws}(:,1));
					maxbeta1cons(draws) = max(slicedraws{draws}(:,1));
					minbeta2cons(draws) = min(slicedraws{draws}(:,2));
					maxbeta2cons(draws) = max(slicedraws{draws}(:,2));

					minbeta1pres(draws) = min(slicedraws{draws}(:,3));
					maxbeta1pres(draws) = max(slicedraws{draws}(:,3));
					minbeta2pres(draws) = min(slicedraws{draws}(:,4));
					maxbeta2pres(draws) = max(slicedraws{draws}(:,4));

					mindelta1(draws) = min(slicedraws{draws}(:,5));
					maxdelta1(draws) = max(slicedraws{draws}(:,5));
					mindelta2(draws) = min(slicedraws{draws}(:,6));
					maxdelta2(draws) = max(slicedraws{draws}(:,6));

					minrho(draws) = min(slicedraws{draws}(:,7));
					maxrho(draws) = max(slicedraws{draws}(:,7));
				elseif model_spec == 4,
					minbeta1cons(draws) = min(slicedraws{draws}(:,1));
					maxbeta1cons(draws) = max(slicedraws{draws}(:,1));
					minbeta2cons(draws) = min(slicedraws{draws}(:,2));
					maxbeta2cons(draws) = max(slicedraws{draws}(:,2));

					minbeta1size(draws) = min(slicedraws{draws}(:,3));
					maxbeta1size(draws) = max(slicedraws{draws}(:,3));
					minbeta2size(draws) = min(slicedraws{draws}(:,4));
					maxbeta2size(draws) = max(slicedraws{draws}(:,4));

					minbeta1pres(draws) = min(slicedraws{draws}(:,5));
					maxbeta1pres(draws) = max(slicedraws{draws}(:,5));
					minbeta2pres(draws) = min(slicedraws{draws}(:,6));
					maxbeta2pres(draws) = max(slicedraws{draws}(:,6));

					mindelta1(draws) = min(slicedraws{draws}(:,7));
					maxdelta1(draws) = max(slicedraws{draws}(:,7));
					mindelta2(draws) = min(slicedraws{draws}(:,8));
					maxdelta2(draws) = max(slicedraws{draws}(:,8));

					minrho(draws) = min(slicedraws{draws}(:,9));
					maxrho(draws) = max(slicedraws{draws}(:,9));					
				end;
			else,
				slicedraws{draws} = -Inf*ones(1,size(x0,2));
				if model_spec == 1,
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
				elseif model_spec == 2,
					minbeta1cons(draws) = Inf;
					maxbeta1cons(draws) = -Inf;
					minbeta2cons(draws) = Inf;
					maxbeta2cons(draws) = -Inf;

					minbeta1pres(draws) = Inf;
					maxbeta1pres(draws) = -Inf;
					minbeta2pres(draws) = Inf;
					maxbeta2pres(draws) = -Inf;

					mindelta1(draws) = Inf;
					maxdelta1(draws) = -Inf;
					mindelta2(draws) = Inf;
					maxdelta2(draws) = -Inf;

					minrho(draws) = Inf;
					maxrho(draws) = -Inf;
				elseif model_spec == 4,
					minbeta1cons(draws) = Inf;
					maxbeta1cons(draws) = -Inf;
					minbeta2cons(draws) = Inf;
					maxbeta2cons(draws) = -Inf;

					minbeta1size(draws) = Inf;
					maxbeta1size(draws) = -Inf;
					minbeta2size(draws) = Inf;
					maxbeta2size(draws) = -Inf;

					minbeta1pres(draws) = Inf;
					maxbeta1pres(draws) = -Inf;
					minbeta2pres(draws) = Inf;
					maxbeta2pres(draws) = -Inf;

					mindelta1(draws) = Inf;
					maxdelta1(draws) = -Inf;
					mindelta2(draws) = Inf;
					maxdelta2(draws) = -Inf;

					minrho(draws) = Inf;
					maxrho(draws) = -Inf;					
				end;				
			end;

		fprintf(1, '%2.0f ... ', draws);
		if mod(draws,10) == 0,
			fprintf(1, '\n');
		end;
	end;
	
	delete(parpoollocal)
	
	% compute credible set (meaning a set that contains the identified set with specified probability)
	% note that if empty identified set at given mu, then count as length 0, and [Inf -Inf] identified set
		if model_spec == 1,
			variables_for_plots = {'beta1', 'beta2', 'delta1', 'delta2', 'rho'}
		elseif model_spec == 2,
			variables_for_plots = {'beta1cons', 'beta2cons', 'beta1pres', 'beta2pres', 'delta1', 'delta2', 'rho'}
		elseif model_spec == 4,
			variables_for_plots = {'beta1cons', 'beta2cons', 'beta1size', 'beta2size', 'beta1pres', 'beta2pres', 'delta1', 'delta2', 'rho'}
		end;
	
		for parm = 1:size(variables_for_plots, 2),
			eval(['mincred_start = min' variables_for_plots{parm}  '(numdraws_extra);']);
			eval(['maxcred_start = max' variables_for_plots{parm}  '(numdraws_extra);']);
			eval(['mincred_bnd = min' variables_for_plots{parm}  ';']);
			eval(['maxcred_bnd = max' variables_for_plots{parm} ';']);
				
			for bound = 0:50000,
				
				credset{parm}(1) = mincred_start - bound/1000;
				credset{parm}(2) = maxcred_start + bound/1000;

				credset_bayescover_comp(1:numdraws) = 0;
				for draws = 1:numdraws,
					if credset{parm}(1) <= mincred_bnd(draws) && credset{parm}(2) >= maxcred_bnd(draws),
						credset_bayescover_comp(draws) = 1;
					end;
				end;
				if (mean(credset_bayescover_comp) >= cred_level-0.005 && mean(credset_bayescover_comp) <= cred_level+0.005) || mean(credset_bayescover_comp) > cred_level+0.005,
					break;
				end;
			end;
			fprintf(1, 'Parm: %2.0f Bayes credibility: %3.2f ... Bayes cred set: %3.2f %3.2f \n', parm, mean(credset_bayescover_comp), credset{parm});
		end;

	try
		close(cur_status);
	catch err
	end
	
	numgridpts = 3000;
	grid = linspace(-30,30,numgridpts);
	inidents(1:size(grid,2)) = 0;
	good_draws_indices = find(good_draws_storage == 1);

	if model_spec == 1,
		variables_for_plots = {'beta1', 'beta2', 'delta1', 'delta2', 'rho'}

		% marginal posterior "curves"
		for var_i = 1:5,
			figure(var_i);
			varname = [variables_for_plots{var_i}];
			inidents(1:size(grid,2)) = 0;
			minaxes_plot = Inf;
			maxaxes_plot = -Inf;
			for draws = good_draws_indices,
				eval(['lowerbnd = min' varname '(draws);']);
				eval(['upperbnd = max' varname '(draws);']);
				minaxes_plot = min(minaxes_plot, lowerbnd);
				maxaxes_plot = max(maxaxes_plot, upperbnd);
				gridlb = max(find(grid <= lowerbnd));
				gridub = min(find(grid >= upperbnd));
				inidents(gridlb:gridub) = inidents(gridlb:gridub) + 1;
			end;
			inidents = inidents/numdraws;
			minaxes_plot_store{var_i} = min(minaxes_plot,credset{var_i}(1)) - 2;
			maxaxes_plot_store{var_i} = max(maxaxes_plot,credset{var_i}(2)) + 2;

			if mod(var_i, 2) == 0,
				minaxes_plot = min(minaxes_plot_store{var_i}, minaxes_plot_store{var_i-1});
				maxaxes_plot = max(maxaxes_plot_store{var_i}, maxaxes_plot_store{var_i-1});
			end;
			axis([minaxes_plot maxaxes_plot 0 1.1]);

			hold on;
			
			% posterior probability of theta in the identified set
			plot(grid, inidents, 'color',[0 0 0]);
			hold on;
			plot(credset{var_i}(1), 0, 'o','color', [0 0 0]);
			hold on;
			plot(credset{var_i}(2), 0, 'o','color', [0 0 0]);
			hold on;
		end;

		% joint posterior contour
		centers={-1:0.1:3 -3:0.1:0};
		for draws = good_draws_indices,
			P{draws}=histproj_fixedgrid([slicedraws{draws}(:,1) slicedraws{draws}(:,3)],centers,0.02);
		end;
		Psum = P{good_draws_indices(1)};
		for draws = good_draws_indices(2:length(good_draws_indices)),
			Psum = Psum + P{draws};
		end;
		Psum = Psum/numdraws;
		figure(6);
		C = contour(centers{1}, centers{2}, Psum);
		colorbar
		xlabel('\beta_{LCC}^{cons}');
		ylabel('\Delta_{LCC}');
		
		max_figs = 6;
	elseif model_spec == 2,
		variables_for_plots = {'beta1cons', 'beta2cons', 'beta1pres', 'beta2pres', 'delta1', 'delta2', 'rho'}

		% marginal posterior "curves"
		for var_i = 1:7,
			if var_i == 1 || var_i == 3 || var_i == 5 || var_i == 7,
				figure(var_i);
			end;
			varname = [variables_for_plots{var_i}];
			inidents(1:size(grid,2)) = 0;
			minaxes_plot = Inf;
			maxaxes_plot = -Inf;
			for draws = good_draws_indices,
				eval(['lowerbnd = min' varname '(draws);']);
				eval(['upperbnd = max' varname '(draws);']);
				minaxes_plot = min(minaxes_plot, lowerbnd);
				maxaxes_plot = max(maxaxes_plot, upperbnd);
				gridlb = max(find(grid <= lowerbnd));
				gridub = min(find(grid >= upperbnd));
				inidents(gridlb:gridub) = inidents(gridlb:gridub) + 1;
			end;
			inidents = inidents/numdraws;
			minaxes_plot_store{var_i} = min(minaxes_plot,credset{var_i}(1)) - 2;
			maxaxes_plot_store{var_i} = max(maxaxes_plot,credset{var_i}(2)) + 2;

			if mod(var_i, 2) == 0,
				minaxes_plot = min(minaxes_plot_store{var_i}, minaxes_plot_store{var_i-1});
				maxaxes_plot = max(maxaxes_plot_store{var_i}, maxaxes_plot_store{var_i-1});
			end;
			axis([minaxes_plot maxaxes_plot 0 1.1]);

			hold on;
			
			% posterior probability of theta in the identified set
			if var_i == 1 || var_i == 3 || var_i == 5 || var_i == 7,
				plot1 = plot(grid, inidents, 'color',[0 0 1]);
				hold on;
				plot(credset{var_i}(1), 0, 'o','color', [0 0 1]);
				hold on;
				plot(credset{var_i}(2), 0, 'o','color', [0 0 1]);
				hold on;
			else,
				plot2 = plot(grid, inidents, 'color',[0 1 0], 'LineStyle', '--');
				hold on;
				plot(credset{var_i}(1), 0, 'o','color', [0 1 0]);
				hold on;
				plot(credset{var_i}(2), 0, 'o','color', [0 1 0]);
				hold on;
			end;
			
			if var_i == 2,
				legend([plot1, plot2], '\beta_{LCC}^{cons}', '\beta_{OA}^{cons}');
			elseif var_i == 4,
				legend([plot1, plot2], '\beta_{LCC}^{pres}', '\beta_{OA}^{pres}');			
			elseif var_i == 6,
				legend([plot1, plot2], '\Delta_{LCC}', '\Delta_{OA}');			
			end;
			hold on;
		end;

		% joint posterior contour for various pairs of parameters
		for var_i = 8:11,

			if var_i == 8,
				vars = [5 6];
				centers={-3:0.1:0 -3:0.1:0};
			elseif var_i == 9,
				vars = [3 4];
				centers={0:0.1:5 0:0.1:1};
			elseif var_i == 10,
				vars = [1 5];
				centers={-2:0.1:2 -3:0.1:0};
			elseif var_i == 11,
				vars = [2 6];
				centers={0:0.1:2 -3:0.1:0};
			end;

			for draws = good_draws_indices,
				P{draws}=histproj_fixedgrid([slicedraws{draws}(:,vars(1)) slicedraws{draws}(:,vars(2))],centers,0.02);
			end;
			Psum = P{good_draws_indices(1)};
			for draws = good_draws_indices(2:length(good_draws_indices)),
				Psum = Psum + P{draws};
			end;
			Psum = Psum/numdraws;
			figure(var_i);
			C = contour(centers{1}, centers{2}, Psum);
			colorbar
			if var_i == 8,
				xlabel('\Delta_{LCC}');
				ylabel('\Delta_{OA}');
			elseif var_i == 9,
				xlabel('\beta_{LCC}^{pres}');
				ylabel('\beta_{OA}^{pres}');
			elseif var_i == 10,
				xlabel('\beta_{LCC}^{cons}');
				ylabel('\Delta_{LCC}');
			elseif var_i == 11,
				xlabel('\beta_{OA}^{cons}');
				ylabel('\Delta_{OA}');
			end;
			
		end;
		
		max_figs = 11;
	elseif model_spec == 4,
		variables_for_plots = {'beta1cons', 'beta2cons', 'beta1size', 'beta2size', 'beta1pres', 'beta2pres', 'delta1', 'delta2', 'rho'}

		% marginal posterior "curves"
		for var_i = 1:9,
			if var_i == 1 || var_i == 3 || var_i == 5 || var_i == 7 || var_i == 9,
				figure(var_i);
			end;
			varname = [variables_for_plots{var_i}];
			inidents(1:size(grid,2)) = 0;
			minaxes_plot = Inf;
			maxaxes_plot = -Inf;
			for draws = good_draws_indices,
				eval(['lowerbnd = min' varname '(draws);']);
				eval(['upperbnd = max' varname '(draws);']);
				minaxes_plot = min(minaxes_plot, lowerbnd);
				maxaxes_plot = max(maxaxes_plot, upperbnd);
				gridlb = max(find(grid <= lowerbnd));
				gridub = min(find(grid >= upperbnd));
				inidents(gridlb:gridub) = inidents(gridlb:gridub) + 1;
			end;
			inidents = inidents/numdraws;
			minaxes_plot_store{var_i} = min(minaxes_plot,credset{var_i}(1)) - 2;
			maxaxes_plot_store{var_i} = max(maxaxes_plot,credset{var_i}(2)) + 2;

			if mod(var_i, 2) == 0,
				minaxes_plot = min(minaxes_plot_store{var_i}, minaxes_plot_store{var_i-1});
				maxaxes_plot = max(maxaxes_plot_store{var_i}, maxaxes_plot_store{var_i-1});
			end;
			axis([minaxes_plot maxaxes_plot 0 1.1]);

			hold on;
			
			% posterior probability of theta in the identified set
			if var_i == 1 || var_i == 3 || var_i == 5 || var_i == 7 || var_i == 9,
				plot1 = plot(grid, inidents, 'color',[0 0 1]);
				hold on;
				plot(credset{var_i}(1), 0, 'o','color', [0 0 1]);
				hold on;
				plot(credset{var_i}(2), 0, 'o','color', [0 0 1]);
				hold on;
			else,
				plot2 = plot(grid, inidents, 'color',[0 1 0], 'LineStyle', '--');
				hold on;
				plot(credset{var_i}(1), 0, 'o','color', [0 1 0]);
				hold on;
				plot(credset{var_i}(2), 0, 'o','color', [0 1 0]);
				hold on;
			end;
			
			if var_i == 2,
				legend([plot1, plot2], '\beta_{LCC}^{cons}', '\beta_{OA}^{cons}');
			elseif var_i == 4,
				legend([plot1, plot2], '\beta_{LCC}^{size}', '\beta_{OA}^{size}');
			elseif var_i == 6,
				legend([plot1, plot2], '\beta_{LCC}^{pres}', '\beta_{OA}^{pres}');			
			elseif var_i == 8,
				legend([plot1, plot2], '\Delta_{LCC}', '\Delta_{OA}');			
			end;
			hold on;
		end;

		max_figs = 9;
	end;
	
	toc;
	
	for var_i = 1:max_figs,
		figure(var_i);
		set(gca, 'LooseInset', get(gca, 'TightInset'));
		set(gcf, 'PaperPosition', [0 0 5 5]);
		set(gcf, 'PaperSize', [5 5]);
		print('-dpdf',strcat('output/empirical_game', int2str(model_spec), '_', int2str(var_i), '.pdf'))
		clf(var_i)
	end;
	
	save(strcat('output/empiricalgamefigs_workspace',int2str(model_spec),'.mat'))
	
end;

zip('output/empiricalgamefigs', {'output/empirical_game*.pdf'});

diary off