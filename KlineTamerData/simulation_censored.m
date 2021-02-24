% Replication files for "Bayesian inference in a class of partially identified models" by Brendan Kline and Elie Tamer
% Simulation for interval-censored regression

clear variables

% parameters of the simulation (change to get true identified set)
n = 2000
total_reps = 250

diary(strcat('output/simulation_censored_', int2str(n), '_diary.txt'))

RandStream.setGlobalStream(RandStream('mt19937ar','Seed', 1));

% parameters of the simulation
	n
	total_reps
	numdraws = 1000;
	numgridpts = 500;
	grid1 = linspace(-13,12,numgridpts);
	grid2 = linspace(-3.5,3.5,numgridpts);
	grid3 = linspace(-2.5,5.0,numgridpts);
	grid4 = linspace(-3.5,6.5,numgridpts);
	credset_cover(1:total_reps) = 0;
	color_list = jet(total_reps);
	cred_level = 0.95;

	figure(4);
	axis([1.0 5.5 0 1.1]);
	hold on;
	figure(8);
	hAxes = axes('DataAspectRatio',[1 1 1], 'XLim',[0.85 1], 'YLim',[0 eps]);       
	hold on;

for reps = 1:total_reps,

	tic
	x = mvnrnd([1 1 1], [1 0.3 0.3; 0.3 1 0.3; 0.3 0.3 1], n);
	u = mvnrnd(0,0.1,n);
	y = -1 + 1*x(:,1) + 2*x(:,2) + 3*x(:,3) + u;
	y_upper = ceil(y);
	y_lower = floor(y);
	momentfn(1:n, 1:19) = 0;
	
	% moment functions
		momentfn(:,1) = y_lower(:)*1;
		momentfn(:,2) = x(:,1);
		momentfn(:,3) = x(:,2);
		momentfn(:,4) = x(:,3);

		momentfn(:,5) = y_lower(:).*x(:,1).^2;	
		momentfn(:,6) = x(:,1).^2;
		momentfn(:,7) = x(:,1).^3;
		momentfn(:,8) = x(:,2).*x(:,1).^2;
		momentfn(:,9) = x(:,3).*x(:,1).^2;

		momentfn(:,10) = y_lower(:).*x(:,2).^2;
		momentfn(:,11) = x(:,2).^2;
		momentfn(:,12) = x(:,1).*x(:,2).^2;
		momentfn(:,13) = x(:,2).^3;
		momentfn(:,14) = x(:,3).*x(:,2).^2;
	
		momentfn(:,15) = y_lower(:).*x(:,3).^2;
		momentfn(:,16) = x(:,3).^2;
		momentfn(:,17) = x(:,1).*x(:,3).^2;
		momentfn(:,18) = x(:,2).*x(:,3).^2;
		momentfn(:,19) = x(:,3).^3;
		
	% parameters of the posterior
		mu_n = mean(momentfn);
		Sigma_n = cov(momentfn);
	
	% draw from the posterior for mu	
		mupost = mvnrnd(mu_n, Sigma_n/n, numdraws);
		mupost(numdraws+1,:) = mu_n';
		
	% compute posterior probabilities over the identified set
		inidents4(1:numgridpts) = 0;
		beta4_contained_in_pos = 0;
	
		for draws = 1:numdraws+1,
			% "objective function"
				mu_matrix = [-1 -mupost(draws,2) -mupost(draws,3) -mupost(draws,4);
						1 mupost(draws,2) mupost(draws,3) mupost(draws,4); 
						-mupost(draws,6) -mupost(draws,7) -mupost(draws,8) -mupost(draws,9);
						mupost(draws,6)  mupost(draws,7)  mupost(draws,8)  mupost(draws,9);
						-mupost(draws,11) -mupost(draws,12) -mupost(draws,13) -mupost(draws,14);
						mupost(draws,11)  mupost(draws,12)  mupost(draws,13)  mupost(draws,14);
						-mupost(draws,16) -mupost(draws,17) -mupost(draws,18) -mupost(draws,19);
						mupost(draws,16)  mupost(draws,17)  mupost(draws,18)  mupost(draws,19)];

				b_matrix = [-mupost(draws,1);
					mupost(draws,1)+1;
					-mupost(draws,5);
					mupost(draws,5)+mupost(draws,6);
					-mupost(draws,10);
					mupost(draws,10)+mupost(draws,11);
					-mupost(draws,15);
					mupost(draws,15)+mupost(draws,16)];

				% since the identified set is convex, a ``marginal'' identified set is an interval
					options = optimset('Display', 'off');
					minbeta4 = linprog([0 0 0 1],mu_matrix,b_matrix, [],[],[],[],[], options);
					minbeta4_store(draws) = minbeta4(4);
					maxbeta4 = linprog(-[0 0 0 1],mu_matrix,b_matrix, [],[],[],[],[], options);
					maxbeta4_store(draws) = maxbeta4(4);
					
					if draws <= numdraws,
						inidents4(min(find(grid4>=minbeta4_store(draws))):max(find(grid4<=maxbeta4_store(draws)))) = inidents4(min(find(grid4>=minbeta4_store(draws))):max(find(grid4<=maxbeta4_store(draws)))) + 1;
						beta4_contained_in_pos = beta4_contained_in_pos + 1*(minbeta4_store(draws) >= 0);
					end;
		
		end;	
		inidents4 = inidents4/numdraws;	
		beta4_contained_in_pos = beta4_contained_in_pos/numdraws;
		
	% compute credible set (meaning a set that contains the identified set with specified probability) for beta4
		for bound = 0:1000,
			credset(1) = minbeta4_store(numdraws+1) + 0.25 - bound/1000;
			credset(2) = maxbeta4_store(numdraws+1) - 0.25 + bound/1000;

			credset_bayescover_comp(1:numdraws) = 0;
			for draws = 1:numdraws,
				if credset(1) <= minbeta4_store(draws) && credset(2) >= maxbeta4_store(draws),
					credset_bayescover_comp(draws) = 1;
				end;
			end;
			if (mean(credset_bayescover_comp) >= cred_level-0.005 && mean(credset_bayescover_comp) <= cred_level+0.005) || mean(credset_bayescover_comp) > cred_level+0.005,
				break;
			end;
		end;

		fprintf(1, 'Bayes credibility: %3.2f ... Bayes cred set: [%3.2f %3.2f] \n', mean(credset_bayescover_comp), credset);
		if credset(1) <= 1.84 && 4.16 <= credset(2),
			credset_cover(reps) = 1;
		end;

	% plot various posterior quantities
		if mod(reps,10) == 0,
			figure(4);
			plot(grid4,inidents4,'color',color_list(reps,:));
			hold on;
			plot(credset(1), 0, 'o','color',color_list(reps,:));
			hold on;
			plot(credset(2), 0, 'o','color',color_list(reps,:));
			hold on;
			figure(8);
			plot(beta4_contained_in_pos,0,'.', 'color',color_list(reps,:), 'MarkerSize',10);
			hold on;
		end;
				
	reps
	toc
end;

fprintf(1, 'Frequentist coverage: %3.3f \n', mean(credset_cover(1:total_reps)));
	
figure(4)
set(gca, 'LooseInset', get(gca, 'TightInset'));
set(gcf, 'PaperPosition', [0 0 5 5]);
set(gcf, 'PaperSize', [5 5]);
print('-dpdf',strcat('output/censfig31.pdf'))

figure(8)
set(gca, 'LooseInset', get(gca, 'TightInset'));
set(gcf, 'PaperPosition', [0 0 5 0.5]);
set(gcf, 'PaperSize', [5 0.5]);
print('-dpdf',strcat('output/censfig32.pdf'))

zip('output/censfigs', {'output/censfig3*.pdf'});

diary off