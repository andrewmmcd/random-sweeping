function blocksStruct = initializeBlocks(optInput)
% % work with
% optInput.points;
% optInput.observations;
% optInput.dimensions;
% optInput.regulParam1;
%
% optInput.iterations;
% optInput.accuracy;
% optInput.checker;
% optInput.blockProbability;
% optInput.relaxationParameter;
%
% optInput.lossType;
% optInput.penaltyType
% optInput.algoParam1
% optInput.algoParam2

% assuming each block Bi is diagonal with 0/1 only

switch optInput.lossType
    case 'square'
        blocksStruct.psiFn =  '@(A,b,w)(1/2)*norm(A*w-b)^2';    
        blocksStruct.proxGammaPsi1 = '@(z,b,inputs)(inputs.gamma*b+inputs.lambda*z)/(inputs.gamma+inputs.lambda)';  
        
    case 'hinge'
        blocksStruct.psiFn =  '@(A,b,w)sum(max(ones(size(A,1),1)-(A*w).*b,0))';    
        blocksStruct.proxGammaPsi1 = '@(z,b,inputs)b.*min(z.*b+(inputs.gamma/inputs.lambda),max(z.*b,1))';  
    
    case 'rootSquare'
        blocksStruct.psiFn =  '@(A,b,w)norm(A*w-b)'; % no 1/2
        blocksStruct.proxGammaPsi1 = '@(z,b,inputs)max(1-inputs.gamma/(inputs.lambda*norm(z-b)),0)*(z-b)+b';  
      
end

switch optInput.penaltyType
    case 'L1'
        %blocksStruct.penaltyFn = '@(Bz,inputs)inputs.lambda*sum(sqrt(sum(Bz.^2,1)))'; % sum of l2 norm of columns 
        blocksStruct.penaltyFn = '@(Bz,inputs)inputs.lambda*norm(diag(Bz),1)'; % equivalently the l1 norm of Bz entries as Bz = z
        
        % dimensions of problem
        d = optInput.numDims;
        blocksStruct.d = d;
        
        % number of blocks        
        n = d;
        blocksStruct.n = n; 
        
        % g norms. assuming g1=..=gn for now    
        blocksStruct.gsFn = '@(w)norm(w)'; 
        
        % prox of g norms.  assuming g1=..=gn for now     
        blocksStruct.proxofgsFn = '@(vi,inputs)max(1-inputs.gamma/norm(vi),0)*vi';
        
        % make B operators. assuming diagonal B with 0/1 entries only 
        % trivial for lasso
        B = eye(d);
        %blocksStruct.B = logical(B);        
        blocksStruct.B = B;      
        
    case 'ksup'
        blocksStruct.penaltyFn = '@(Bz,inputs)inputs.lambda*sum(sqrt(sum(Bz.^2,1)))'; % sum of l2 norm of columns
        
        % dimensions of problem
        d = optInput.numDims;
        blocksStruct.d = d;
        
        % number of blocks        
        k = optInput.algoParam1;
        n = nchoosek(d,k);
        blocksStruct.n = n; 
        
        % g norms. assuming g1=..=gn for now    
        blocksStruct.gsFn = '@(w)norm(w)'; 
        
        % prox of g norms.  assuming g1=..=gn for now     
        blocksStruct.proxofgsFn = '@(v,inputs)max(1-inputs.gamma/norm(v),0)*v';
        
        % make B operators. assuming diagonal B with 0/1 entries only 
        B = getKSupGroups(d,k);
        %blocksStruct.B = logical(B);        
        blocksStruct.B = B;    
        
           
    case 'chain'
        blocksStruct.penaltyFn = '@(Bz,inputs)inputs.lambda*sum(sqrt(sum(Bz.^2,1)))'; % sum of l2 norm of columns
        
        % dimensions of problem
        d = optInput.numDims;
        blocksStruct.d = d;
        
        % get groups, make B operators. assuming diagonal B with 0/1 entries only
        gLength = optInput.algoParam1;
        gOverlap = optInput.algoParam2;        
        B = getChainGroups(d,gLength,gOverlap);
        
        % 
        blocksStruct.B = B; 
               
        % number of blocks                
        n = size(B,2);
        blocksStruct.n = n; 
        
        % g norms. assuming g1=..=gn for now    
        blocksStruct.gsFn = '@(w)norm(w)'; 
        
        % prox of g norms.  assuming g1=..=gn for now     
        blocksStruct.proxofgsFn = '@(v,inputs)max(1-inputs.gamma/norm(v),0)*v';
        
end


end

