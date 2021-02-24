function p = bvn( xl, xu, yl, yu, corr )
  % replace bvnfastcdf with any (fast!) bivariate normal CDF algorithm... first argument is point of evaluation, second argument is mean, and third argument is covariance

  p = bvnfastcdf([xu yu], [0 0], [1 corr; corr 1]) - bvnfastcdf([xl yu], [0 0], [1 corr; corr 1]) - bvnfastcdf([xu yl], [0 0], [1 corr; corr 1]) + bvnfastcdf([xl yl], [0 0], [1 corr; corr 1]); 

end
