% sample script for random sweeping algorithm
% see paper: https://arxiv.org/pdf/1603.09273v1.pdf
% example: root lasso regression:  min_w || Xw - y || + lambda ||w||_1

addpath(genpath('functions'))

%% generate data
% params
numPoints = 200;
numDims = 500;
sparsity = 0.80;
noise = 0.10;

% data (normalised)
points = normr(randn(numPoints, numDims));

% regression vector (normalised)
trueWeightVector = randn(numDims,1);
numZero = ceil(sparsity*numDims);
if numZero > 0
    randIx = randperm(numDims,numZero);
    trueWeightVector(randIx) = 0;
end
trueWeightVector = normc(trueWeightVector);

% observations (with noise)
observations = points * trueWeightVector + noise*randn(numPoints,1);

optInput.points = points;
optInput.observations = observations;
optInput.numDims = numDims;

%% setup optimization
optInput.lossType = 'rootSquare';   % ONE OF: 'square', 'rootSquare', 'hinge'
optInput.penaltyType = 'L1';        % ONE OF: 'L1', 'chain', 'ksup'
optInput.experimentType = 'regression';
optInput.algoParam1 = 2;     % parameters for regularizer: chain uses both (length, overlap), ksup only first one (k) 
optInput.algoParam2 = 0;
optInput.computeWinf = 0;    % 1 or 0, if 1 pass vector as Winf to compute distance over time  
optInput.Winf = 0;
optInput.accuracy = 10^(-6);
optInput.explicit = 1;      % output
optInput.checker = 10;      % frequency of output
optInput.blockPctPen = 1;   % percentage of blocks turned on each iteration
optInput.maxObjective = Inf;
optInput.initialization = 'zero';
optInput.regulParam1 = 0.01;     % >0
optInput.regulParam2 = 0.001;    % >0
optInput.regulParam3 = 1.9500;   % in (0,2)
optInput.maxIterations = 100000; 
optInput.minIterations = 10;


%% run optimization
tic;
optOutput = blocksOptimization(optInput);
endTic = toc;


%% summary
foundVector = optOutput.regressionVector;
squareLossError = 0.5*(1/numPoints)*norm(points*foundVector - observations)^2;

fprintf('summary\n');
fprintf('  experiment              : %s\n', optInput.experimentType )
fprintf('  loss                    : %s\n', optInput.lossType)
fprintf('  penalty                 : %s\n', optInput.penaltyType)
fprintf('  algoParam1              : %d\n', optInput.algoParam1)
fprintf('  algoParam2              : %d\n', optInput.algoParam2)
fprintf('  reg param1 (lambda)     : %1.5f\n', optInput.regulParam1)
fprintf('  reg param2 (gamma)      : %1.5f\n', optInput.regulParam2)
fprintf('  reg param3 (mu)         : %1.5f\n', optInput.regulParam3)
fprintf('  iters                   : %d\n',  optOutput.numIters)
fprintf('  timer                   : %1.5f\n',  optOutput.timer)
fprintf('  final objective         : %2.8f\n',  optOutput.costs(end))
fprintf('  tol                     : %e\n', optInput.accuracy)
fprintf('  error (square loss)     : %e\n', squareLossError)


figure
plot(optOutput.costs, 'rx-')
title('objective')



