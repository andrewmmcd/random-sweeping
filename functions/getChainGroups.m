function chainGroups = getChainGroups(numDims,gLength,gOverlap)
%

gLength = min(gLength,numDims); % this is ensured when run as part of blocks algo but is safety net

numGroups = ceil((numDims-gOverlap)/(gLength-gOverlap));
chainGroups = zeros(numDims,numGroups);

ixStart = 1;
ixEnd = gLength;

for i=1:numGroups
    chainGroups(ixStart:ixEnd,i) = ones(ixEnd-ixStart+1,1);
    ixStart = ixStart + gLength - gOverlap;
    ixEnd = min(ixEnd+gLength-gOverlap,numDims);
end


end

