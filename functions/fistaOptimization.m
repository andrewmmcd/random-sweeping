function optOutput = fistaOptimization(optInput)
% solves regularization problem for specific reg param and inputs using FISTA
% currently implements L1 with square loss
% this is called for each regularizer value by regularizerValidation, which chooses the weights by validation
% currently: solves  1/2 || Xw - y ||_2^2 + 1/2 lambda || w ||_1
% X: points, numPoints x dims
% y: observations, numPoints x 1
% expects as input a struct with:
%           points: [numPoints x dims]
%     observations: [numPoints x 1]
%       dimensions: dims
%         lossType: 'square'
%      penaltyType: 'L1'
%       iterations: double
%         accuracy: double
%          checker: int
%      regulParam1: double
     
X               = optInput.points;
y               = optInput.observations;
numDims         = optInput.numDims;
lambda          = optInput.regulParam1;

maxIterations   = optInput.maxIterations;
minIterations   = optInput.minIterations;
accuracy        = optInput.accuracy;
checker         = optInput.checker;
explicit        = optInput.explicit;
istaType        = optInput.istaType;
algoParam       =  optInput.algoParam1;  %k
computeWinf     = optInput.computeWinf ;
Winf            = optInput.Winf; %first time run this is 0. second time it's final regression vector

%Lipschitz = 2 * eigs([X' * X, X' * X; X' * X, X' * X], 1);    % confirm this for simple regression

switch istaType
    case 'ista'
        Lipschitz = eigs([X' * X], 1)/1.99;
    case 'fista'
        Lipschitz = eigs([X' * X], 1);
end

    
switch optInput.lossType
    case 'square'
        loss = @(w)  1/2*norm(X*w - y)^2;
        gradOfLoss = @(w)  X'*(X*w - y);   
end

switch optInput.penaltyType
    case 'L1'
        penalty = @(w) lambda*sum(abs(w));
        proxOfPenalty = @(w, L) sign(w).*max(abs(w) - lambda/L, 0);
    case 'ksup'
        penalty = @(w) lambda*boxNorm(w,0.000000001,1,algoParam);  % FIX THIS TO K SUP
        proxOfPenalty = @(w, L) boxNormProx(w, L, lambda, algoParam );
end

% intialization
fixedPt = zeros(numDims, 1);  % convex combination/line search of previous point and prox point
proxPt  = zeros(numDims, 1);   % proximal point before line search
iter = 0;
curr_obj = loss(fixedPt) + penalty(fixedPt);
costs = curr_obj;
theta = 1;
difference = Inf;
distanceFromFinal = norm(fixedPt-Winf);


if explicit
    if computeWinf        
        fprintf('Computing distance to winf\n')
    end
    fprintf('Iter %7d: | Current obj: %10.4f | Lipschitz: %2.3f | Time: %s\n',...
        iter, curr_obj, Lipschitz, datestr(clock, 0))
end

%iterTiming = [];



while (iter < minIterations) || (iter < maxIterations && difference > accuracy)
    
    %startIteration = tic;
    
    iter = iter + 1;
    
    prevInterPt = proxPt;
    proxPt = proxOfPenalty(fixedPt - (1/Lipschitz) * gradOfLoss(fixedPt),Lipschitz);
    
    
    switch istaType
        case 'ista'
            fixedPt = proxPt;
        case 'fista'
            theta = (sqrt(theta^4+4*theta^2)-theta^2)/2;
            rho = 1-theta+sqrt(1-theta);
            fixedPt = rho*proxPt - (rho-1)*prevInterPt;
    end
    
    
    prev_obj = curr_obj;
    curr_obj = loss(fixedPt) + penalty(fixedPt);
    costs = [costs curr_obj];
    %weights = [weights proxPt];
    if computeWinf
        distanceFromFinal = [distanceFromFinal, norm(fixedPt-Winf)];
    end
    
    %difference = abs(prev_obj - curr_obj);
    %difference = norm(prevInterPt - proxPt);
    difference = norm(prevInterPt - proxPt)/norm(proxPt);
    
    if(mod(iter, checker) == 0) && explicit
        fprintf('Iter %7d: | Current obj: %10.4f | Lipschitz: %2.3f | Time: %s\n',...
            iter, curr_obj, Lipschitz, datestr(clock, 0))
    end
    
    
        
    %iterTiming = [iterTiming toc(startIteration)];
    
    
end

%fprintf('%d x %d, ista, iters: %d, avg iteration time: %1.10f\n', size(X,1), size(X,2),iter,mean(iterTiming)')




if explicit
    fprintf('Iter %7d: | Current obj: %10.4f | Lipschitz: %2.3f | Time: %s\n\n',...
        iter, curr_obj, Lipschitz, datestr(clock, 0))
end
    
    

optOutput.regressionVector = proxPt;
optOutput.costs = costs;
optOutput.distanceFromFinal = distanceFromFinal;
optOutput.numIters = iter;

end
