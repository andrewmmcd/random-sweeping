function optOutput = blocksOptimization(optInput)
% solves regularized regression/classification problem using random sweeping algorithm
% see paper: https://arxiv.org/pdf/1603.09273v1.pdf
%
% this implementation assumes operators Bi are diagonal with 0/1 only on diagonal
% no sweeping over loss 
% expects as input a struct with:
%             points: [numPoints x dims double]
%       observations: [numPoints x 1 double]
%            numDims: dims
%           lossType: ONE OF: 'square', 'rootSquare', 'hinge'
%        penaltyType: ONE OF: 'L1', 'chain', 'ksup'
%     experimentType: description, ONE OF: 'regression', 'classification'
%         algoParam1: 0      % parameters for regularizer: chain uses both (length, overlap), ksup only first one (k) 
%         algoParam2: 0     
%        computeWinf: 0      % 1 or 0. if 1 it computes    
%               Winf: 0      % if vector is passed as input and computeWinf is on, computes distance each iteration
%           accuracy: 1.0000e-03
%           explicit: ONE OF: 1 OR 0    % flag for output during algorithm
%            checker: 1000              % if explicit, output at this rate
%        blockPctPen: 1                 % strictly positive percentage in (0,1] 
%       blockPctLoss: 1                 % not used in this version
%       maxObjective: Inf
%     initialization: ONE OF: 'zero' or 'random'
%        regulParam1: 0.1000            % regularization parameter >0    
%        regulParam2: 0.0100            % prox parameter >0
%        regulParam3: 1.9500            % DR relaxation parameter in (0,2)
%      maxIterations: 100000 
%      minIterations: 10
          

%% parameters
maxIterations   = optInput.maxIterations;
minIterations   = optInput.minIterations;
maxObjective    = optInput.maxObjective;
accuracy        = optInput.accuracy;
checker         = optInput.checker;
explicit        = optInput.explicit;
blockPctPen     = optInput.blockPctPen;
lambda          = optInput.regulParam1;
gamma           = optInput.regulParam2;
mu              = optInput.regulParam3;
initialization  = optInput.initialization;
algoParam1      = optInput.algoParam1;
algoParam2      = optInput.algoParam2;
computeWinf     = optInput.computeWinf;
Winf            = optInput.Winf; 

% data
A = optInput.points; 
b = optInput.observations;
d = optInput.numDims;
m = size(A,1);

%% initialize block operators. 
blocksStruct = initializeBlocks(optInput);

n = blocksStruct.n; % num blocks
% for diag 0/1 only we get dxn matrix  b1... bn.  means Bi = diag(bi) are self adjoint, Bi*Bi' = Bi
B = blocksStruct.B;

% any inputs required to compute penalty, loss need to be passed in structure 'inputs'
% currently handles l1, ksup, lgl chain penalties
inputs.lambda   = lambda;
inputs.gamma 	= gamma;
inputs.k        = algoParam1;


fullPsi  = str2func(blocksStruct.psiFn);        %@(A,b,w)

fullPenalty = str2func(blocksStruct.penaltyFn); %@(w,lambda)
penalty1 = @(Bz) fullPenalty(Bz,inputs);          %@(Bz)

fullProxGammaPsi1 = str2func(blocksStruct.proxGammaPsi1);   %@(z,b,inputs)
proxGammaPsi1 = @(z) fullProxGammaPsi1(z,b,inputs);         %@(z)

fullProxGammaGi = str2func(blocksStruct.proxofgsFn);    %@(v,inputs)
proxGammaGi = @(vi) fullProxGammaGi(vi,inputs);           %@(v)


%% intialization
iter = 0;
numPenBlocks  = ceil(n*blockPctPen);

% initialise variables
switch initialization
    case 'zero'
        Z = zeros(d,n);
        V = zeros(d,n);
        w = zeros(m,1);
        y = zeros(m,1);
    case 'random'
        Z = zeros(d,n).*B;
        V = zeros(d,n).*B;
        w = zeros(m,1);
        y = zeros(m,1);
end

% store [B1v1.. Bnvn] and [B1z1.. Bnzn] for easy updates, and easy computation w=sum(Bv,2)
Bv = B.*V;
Bz = B.*Z;

regressionVector = sum(Bz,2);

% compute C = D^-1 = (I_m + L L*)^-1 = (I + A(sum_{i=1}^n Bi Bi*)A*)^-1
% special case: sum Bi Bi* = sum Bi = sum diag(bi) = diag(sum(bi)) = diag(sum(B,2))
D = (eye(m) + A* diag(sum(B,2)) * A');
Dinv = inv(D);

obj = fullPsi(A,b,regressionVector) + penalty1(Bz);
costs = obj;

distanceFromFinal = norm(regressionVector-Winf);

difference = Inf;
if  explicit
    if computeWinf        
        fprintf('Computing distance to winf\n')
    end
    fprintf('Iter %7d: | Obj: %10.16f | Diff: N/A                | Time: %s\n', iter, obj, datestr(clock, 0))
end


%% iterate

tic
while ((iter < minIterations) || (iter < maxIterations && difference > accuracy)) && (obj < maxObjective)
    iter = iter + 1;
    
    % previous values
    oldV = V;
    oldZ = Z;
    oldy = y;
    oldBv = Bv;
    oldBz = Bz;
    oldObj = obj;
    
    % choose random numVBlocks to turn on
    activeLossBlocks = randperm(n,numPenBlocks);
    
    % common to all Qi, i=1..n
    % same as A'* D^-1 *(A*weightsk -y) but better numerically
    Tk = A'* Dinv * (A*sum(oldBv,2)-oldy);

    
    for activeVix = 1:length(activeLossBlocks)
        i = activeLossBlocks(activeVix);
        % update Qi, zi, vi
        % compute Qi(v1, .. vn, y) = vi - (Bi*) (A*) D^-1(A sum_{j=1}^n Bj vj - y)   *: transpose/adjoint
        Qi = oldV(:,i)-B(:,i).*Tk;        
        Z(:,i) = Qi;
        V(:,i) = oldV(:,i) + mu*( proxGammaGi(2*Z(:,i)-oldV(:,i)) - Z(:,i)) ;  
    end
    
%     if activePenBlocks
        % no sweeping over blocks
        % update Qn+1, w, y
        % Qn+1 = yk + C(A sumjBjvj - yk)
        
        % C^-1 = D, so these two are equivalent
        %Qnp1 = y + C*(A*weightsk-y);
        Qnp1 = oldy + (D\(A*sum(oldBv,2)-oldy));
        
        w = Qnp1;
        y = oldy + mu*(proxGammaPsi1(2*w-oldy) - w);
%     end
    
    
    
    
    % smart update Bivi, Bizi and regressionVector = sum(Bizi)
    Bv(:,activeLossBlocks) = B(:,activeLossBlocks).*V(:,activeLossBlocks);
    Bz(:,activeLossBlocks) = B(:,activeLossBlocks).*Z(:,activeLossBlocks);
    
    regressionVector = regressionVector + sum(Bz(:,activeLossBlocks),2)-sum(oldBz(:,activeLossBlocks),2);
    
    if computeWinf
        distanceFromFinal = [distanceFromFinal, norm(regressionVector-Winf)];
    end
    
    obj = fullPsi(A,b,regressionVector) + penalty1(Bz);
    costs = [costs obj];
    
    
    % CHANGE IN ITERATES
%     if isequal(Z,oldZ) 
%         zkdiff = 0;
%     else
%         zkdiff = norm(Z-oldZ,'fro')/norm(oldZ,'fro');
%     end
%     difference = zkdiff;
    
    
    % CHANGE IN OBJECTIVE
    if isequal(obj,oldObj) 
        objDiff = 0;
    else
        if oldObj > 0
            objDiff = abs(obj-oldObj)/oldObj;
        else
            objDiff = Inf;
        end
    end
    difference = objDiff;

    
    if(mod(iter, checker) == 0) && explicit
        fprintf('Iter %7d: | Obj: %10.16f | Diff: %10.16f | Time: %s\n', iter, obj, difference, datestr(clock, 0))
    end
    
end
endTic = toc;

if  explicit
    fprintf('Iter %7d: | Obj: %10.16f | Diff: %10.16f | Time: %s\n', iter, obj, difference, datestr(clock, 0))
end

%% done

optOutput.regressionVector = regressionVector;
optOutput.costs = costs;
optOutput.distanceFromFinal = distanceFromFinal;
optOutput.numIters = iter;
optOutput.timer = endTic;


end




