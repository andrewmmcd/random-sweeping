function loss = bigPsiSquare(eta,beta)
% computes Psi(eta) = sum_i psii(etai) = sum_i li(etai,betai) in case of li = 1/2 euclidean norm squared

loss = (1/2)*norm(eta-beta,2)^2;

end

