function criterion_fn = criterion_fn(x, tol, sign_restrict)
% criterion function for game model

dimx = size(x, 2);
p11 = x(dimx - 3);
p10 = x(dimx - 2);
p01 = x(dimx - 1);
p00 = x(dimx);

if sign_restrict,
	% this rules out the following: any positive deltas, invalid correlations, and negative betas
		if x(3) > 0 || x(4) > 0 || x(5) < 0 || x(5) > 1 || x(1) < 0 || x(2) < 0,
			criterion_fn = 10000;
			return;
		end;
else,
	% this rules out the following: any positive deltas, invalid correlations, but NOT negative betas
		if x(3) > 0 || x(4) > 0 || x(5) < 0 || x(5) > 1,
			criterion_fn = 10000;
			return;
		end;
end;

% compute the criterion function
	sigma = [1 x(5); x(5) 1];

	model_11 = mvncdf2(sigma, [-x(1)-x(3)  -x(2)-x(4)], [Inf Inf]);
		criterion_fn = (p11 - model_11)^2;
		if check_larger(criterion_fn,tol),
			return;
		end;

	model_00 = mvncdf2(sigma, [-Inf -Inf], [-x(1)  -x(2)]);
		criterion_fn = criterion_fn + (p00 - model_00)^2;
		if check_larger(criterion_fn,tol),
			return;
		end;

	model_01_uniq = mvncdf2(sigma, [-Inf -x(2)], [-x(1)  Inf]) + mvncdf2(sigma, [-x(1)   -x(2)-x(4)], [-x(1)-x(3)  Inf]);
	model_multiple = mvncdf2(sigma, [-x(1) -x(2)], [-x(1)-x(3)  -x(2)-x(4)]);
	if model_multiple > 0,
		s = (p01-model_01_uniq)/model_multiple;
	else,
		s = 0;
	end;
	model_01 = model_01_uniq+s*model_multiple;
		criterion_fn = criterion_fn + (p01 - model_01)^2 + abs(s)*(1*(s<0)) + (s-1)*(1*(s>1));	
		if check_larger(criterion_fn,tol),
			return;
		end;

	model_10 = 1- model_11 - model_00 - model_01; 
		criterion_fn = criterion_fn + (p10 - model_10)^2;

end

function p = mvncdf2(sigma, LB, UB)

	p = bvn( LB(1), UB(1), LB(2), UB(2), sigma(1,2) );

end

function isLarger = check_larger(val, tolerance)

	if val > tolerance,
		isLarger = true;
	else
		isLarger = false;
	end;
end