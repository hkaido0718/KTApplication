% Replication files for "Bayesian inference in a class of partially identified models" by Brendan Kline and Elie Tamer
% Simulation for an interval identified parameter

diary('output/simulation_interval_diary.txt')

clear variables

for overall_rep = 1:3,
	RandStream.setGlobalStream(RandStream('mt19937ar','Seed', 1));
	numdraws = 5000;

	partially_identified = 0;
	if overall_rep == 1,
		partially_identified = 1;
	end;

	if partially_identified,
		mu = [0 1];
		sigma = [1 0;
			0 1];
		d = 2;
		figure(1);
		axis([-0.5 1.5 0 1.1]);
		hold on;
	else,
		mu = [0 0];
		sigma = [1 0;
			0 1];
		d = 2;
		figure(1);
		axis([-0.5 0.5 0 1.1]);
		hold on;
	end;

	n = 500;
	numgridpts = 800;
	grid = linspace(-0.5,1.5,numgridpts);
	total_reps = 500;
	credset_cover(1:total_reps) = 0;
	color_list = jet(total_reps);
	cred_level = 0.95;
	
	figure(2);
	hAxes = axes('DataAspectRatio',[1 1 1], 'XLim',[0 1], 'YLim',[0 eps]);       
	hold on;

	overall = tic;
	for reps = 1:total_reps,
		% simulate the data
			x = mvnrnd(mu,sigma,n);

		% parameters of the posterior
			mu_n = mean(x);
			Sigma_n = cov(x);

		% draw from the (approximate) posterior for mu	
			normdraw = mvnrnd([0 0], Sigma_n/n, numdraws);
			mupost = repmat(mu_n,[numdraws 1]) + normdraw;

		% "draw" from the posterior for the identified set for theta, which is characterized by mu_1 <= theta <= mu_2
		% note that if empty identified set at given mu, then count as length 0, and [Inf -Inf] identified set
			identset(1:numdraws,1:2) = 0;
			identset_wlength(1:numdraws,1:3) = 0;
			for draws = 1:numdraws,
				if mupost(draws,1) <= mupost(draws,2),
					identset(draws,:) = mupost(draws,:);
					identset_wlength(draws,:) = [mupost(draws,:) mupost(draws,2)-mupost(draws,1)];
				else,
					identset(draws,:) = [Inf -Inf];
					identset_wlength(draws,:) = [Inf -Inf 0];
				end;
			end;

		% compute quantities related to the posterior for the identified set for theta
			% probability that the identified set is empty
			probempty = 0;
			for draws = 1:numdraws,
				if identset(draws,:) == [Inf -Inf],
					probempty = probempty + 1;
				end;
			end;
			probempty = probempty/numdraws;

			% compute posterior probability at various points
			inidents(1:size(grid,2)) = 0;
			for gridpt = 1:size(grid,2),
				inidents(gridpt) = 0;
				for draws = 1:numdraws,
					if identset(draws,1) <= grid(gridpt) && grid(gridpt) <= identset(draws,2),
						inidents(gridpt) = inidents(gridpt) + 1;
					end;
				end;
				inidents(gridpt) = inidents(gridpt)/numdraws;
			end;
			
		% compute credible set (meaning a set that contains the identified set with specified probability)
			if probempty > cred_level,
				credset = [Inf -Inf];
			else,
				if probempty < 0.25,
					lbound = 0;
					bound_denom = 2000;
				else,
					lbound = -1000;
					bound_denom = 3000;
				end;

				for bound = lbound:1000,
					credset(1) = mu_n(1) - bound/bound_denom;
					credset(2) = mu_n(2) + bound/bound_denom;

					credset_bayescover_comp(1:numdraws) = 0;
					for draws = 1:numdraws,
						if (credset(1) <= identset(draws,1) && credset(2) >= identset(draws,2)),
							credset_bayescover_comp(draws) = 1;
						end;
					end;
					
					if (mean(credset_bayescover_comp) >= cred_level-0.005 && mean(credset_bayescover_comp) <= cred_level+0.005) || mean(credset_bayescover_comp) > cred_level+0.005,
						break;
					end;
				end;
			end;

			if credset(1) <= mu(1) && mu(2) <= credset(2),
				credset_cover(reps) = 1;
			end;	
			
			credset_bayescover_comp(1:numdraws) = 0;
			for draws = 1:numdraws,
				if credset(1) <= identset(draws,1) && credset(2) >= identset(draws,2),
					credset_bayescover_comp(draws) = 1;
				end;
			end;

			fprintf(1, 'Rep: %4.0f/%4.0f, Bayes credibility: %3.2f ... Bayes cred set: %3.2f %3.2f, Did cover?: %1.0f, Cumulative Frequentist Coverage: %2.3f \n', reps, total_reps, mean(credset_bayescover_comp), credset, credset_cover(reps), mean(credset_cover(1:reps)));
					
		% plot various posterior quantities
			if mod(reps,10) == 0,
				figure(1);
				if overall_rep <= 2,
					plot(grid, inidents, 'color',color_list(reps,:));
					hold on;
					plot(credset(1), 0, 'o','color',color_list(reps,:));
					hold on;
					plot(credset(2), 0, 'o','color',color_list(reps,:));
				elseif overall_rep == 3,
					plot(grid, inidents/max(inidents), 'color',color_list(reps,:));
					hold on;
					plot(mean(mu_n), max(inidents/max(inidents)), 'o','color',color_list(reps,:));
				end;
				hold on;
				
				figure(2);
				plot(1-probempty,0,'.', 'color',color_list(reps,:), 'MarkerSize',10);
				hold on;
			end;		
	end;
	probempty
	mean(credset_cover)
	toc(overall)
	
	figure(1);
	set(gca, 'LooseInset', get(gca, 'TightInset'));
	set(gcf, 'PaperPosition', [0 0 5 5]);
	set(gcf, 'PaperSize', [5 5]);
	print('-dpdf',strcat('output/intfig', int2str(overall_rep) ,'1.pdf'))
	clf(1)
	figure(2);
	set(gca, 'LooseInset', get(gca, 'TightInset'));
	set(gcf, 'PaperPosition', [0 0 5 0.5]);
	set(gcf, 'PaperSize', [5 0.5]);
	print('-dpdf',strcat('output/intfig', int2str(overall_rep) ,'2.pdf'))
	clf(2)
end;

zip('output/intervalfigs', {'output/intfig*.pdf'});

diary off