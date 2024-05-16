% Function that calculates distances from cluster center
function p_dist = distanceFromCenter(cluster_center, X)
    p_dist = zeros(size(X, 1), 1);
    for i = 1:size(X, 1)
        dist = pdist2(cluster_center, X(i, :));
        p_dist(i) = dist;
    end
end
