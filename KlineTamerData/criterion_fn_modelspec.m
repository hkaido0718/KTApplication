function criterion_fn_rtn = criterion_fn_modelspec(x, mu, tol, model_spec)
% criterion function for game model

	if model_spec == 1,
		% parameter is x --- [beta1 beta2 delta1 delta2 rho]
		criterion_fn_rtn = criterion_fn([x mu], tol, 0);
	elseif model_spec == 2,
		% parameter is x --- [beta1cons beta2cons beta1pres beta2pres delta1 delta2 rho]
		criterion_fn_rtn = criterion_fn([x(1) x(2) x(5) x(6) x(7) mu{1,1}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(3) x(2) x(5) x(6) x(7) mu{2,1}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1) x(2)+x(4) x(5) x(6) x(7) mu{1,2}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(3) x(2)+x(4) x(5) x(6) x(7) mu{2,2}], tol, 0);
	elseif model_spec == 4,
		% parameter is x --- [beta1cons beta2cons beta1size beta2size beta1pres beta2pres delta1 delta2 rho]
		criterion_fn_rtn = criterion_fn([x(1) x(2) x(7) x(8) x(9) mu{1,1,1}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(3) x(2)+x(4) x(7) x(8) x(9) mu{1,1,2}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1) x(2)+x(6) x(7) x(8) x(9) mu{1,2,1}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(3) x(2)+x(4)+x(6) x(7) x(8) x(9) mu{1,2,2}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(5) x(2) x(7) x(8) x(9) mu{2,1,1}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(3)+x(5) x(2)+x(4) x(7) x(8) x(9) mu{2,1,2}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(5) x(2)+x(6) x(7) x(8) x(9) mu{2,2,1}], tol, 0);
		if check_larger(criterion_fn_rtn,tol),
			return;
		end;

		criterion_fn_rtn = criterion_fn_rtn + criterion_fn([x(1)+x(3)+x(5) x(2)+x(4)+x(6) x(7) x(8) x(9) mu{2,2,2}], tol, 0);

	end;

end

function isLarger = check_larger(val, tolerance)

	if val > tolerance,
		isLarger = true;
	else
		isLarger = false;
	end;
end