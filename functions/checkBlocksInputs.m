function inputCheck = checkBlocksInputs(params)
% checks inputs to biclustering are acceptable

%ticCheckInputs = displayTimer(params, -1, 'Validating inputs', 0);
%[inputCheck, inputValidateMsg] = checkInputs(params);

inputCheck = 0;
checkMsg = '';

switch params.experiment.type
    case 'regression'
        % dimensions
        if  params.data.numPoints < 0 || params.data.numDims < 0 
            inputCheck = 1;
            checkMsg = [checkMsg 'Ensure data dimensions are positive integers.\n'];
        end
        
end

switch params.algo.algoType
    case {'blocks', 'blocksV2'}
        if params.algo.blockPctPen <= 0 || params.algo.blockPctPen > 1 || ...
                params.algo.blockPctLoss <= 0 || params.algo.blockPctLoss > 1
            inputCheck = 1;
            checkMsg = [checkMsg 'Ensure block probabilities in (0,1].\n'];
        end
end

if params.algo.maxIterations < params.algo.minIterations
    inputCheck = 1;
    checkMsg = [checkMsg 'Ensure minIter < maxIter.\n'];
end


% need: train + val + test = 1 
if params.algo.trainPct + params.algo.valPct + params.algo.testPct ~= 1     
    inputCheck = 1;
    checkMsg = [checkMsg 'Ensure minIter < maxIter.\n']; 
end

% and either:  
% 1) tr, val, te > 0  [full validation]
% 2) tr, te>0, val=0  [train, test only]
% 3) tr>0, val=tr=0   [train only]
if ~ ((params.algo.trainPct > 0 && params.algo.valPct > 0 && params.algo.testPct > 0) || ...
      (params.algo.trainPct > 0 && params.algo.valPct == 0 && params.algo.testPct > 0) || ...
      (params.algo.trainPct > 0 && params.algo.valPct == 0 && params.algo.testPct == 0) )
    inputCheck = 1;
    checkMsg = [checkMsg 'Ensure either tr,val,tst>0 or tr,test>0, or tr>0.\n']; 
end


% and 
if params.algo.valPct == 0 && (length(params.algo.param1) > 1 ||  length(params.algo.param2) > 1 ||  length(params.algo.param3) > 1)
    inputCheck = 1;
    checkMsg = [checkMsg 'If no validation (val=0) only one (lambda, gamma, mu) is allowed.\n']; 
end


switch params.experiment.penaltyType
    case 'ksup'
        if params.data.numDims < params.experiment.algoParam1
           inputCheck = 1;
            checkMsg = [checkMsg 'Need k <= d for ksup.\n'];  
        end
    case 'chain'
        if params.data.numDims < params.experiment.algoParam1 || params.experiment.algoParam1 <= params.experiment.algoParam2
            inputCheck = 1;
            checkMsg = [checkMsg 'Need gOverlap < gLength < d for chain LGL.\n'];  
        end
end


%displayTimer(params, ticCheckInputs, 'Validating inputs',  0);

if inputCheck >0
    fprintf(['Error: ' checkMsg '\n'])
    return
end


end

