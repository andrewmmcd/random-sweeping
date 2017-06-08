function loss = bigPsiHinge(eta,beta)
% computes Psi(eta) = sum_i psii(etai) = sum_i li(etai,betai) in case of li = hinge loss
% @(A,b,w)sum(max(ones(size(A,1),1)-(A*w).*b,0))

loss = sum(max(ones(size(eta,1),1)-eta.*beta,0));

end

