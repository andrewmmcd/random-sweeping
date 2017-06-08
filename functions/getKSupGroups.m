function ksupGroups = getKSupGroups(n,k)
% produce ksup groups
if k == 0
    ksupGroups = zeros(n,1);
elseif k==n
    ksupGroups = ones(n,1);    
else
    % n>k>0
    ksupGroups = [mergeGroups(1,getKSupGroups(n-1,k-1)) mergeGroups(0,getKSupGroups(n-1,k))];        
end

end

function newG = mergeGroups(i,G)
    numGroups = size(G,2);
    newG = [repmat(i,1,numGroups); G];    
end
