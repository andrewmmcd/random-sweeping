function penalty = bigPhiL2(Vs,inputs)
% computes Phi([v1,...,vm]) = lambda sum_j fj(vj) in case fj = l2 norm

penalty = inputs.lambda * sum(sqrt(sum(Vs.^2,1)));

end

