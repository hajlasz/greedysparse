addpath 'suite_sparse_api'
index = ssget; % gets the struct of the indices of the matrices
f = find(index.numerical_symmetry == 1 & index.isReal);

% sort the indices in f based on number of rows, low to high
% y is number of rows
% j is the index in f of the matrix (not in ssget)
[y,j] = sort(index.nrows(f));

f = f(j);
c = 933; % for the purposes of skipping matrices that fail to load

% downloading the matrices
for i = f(c:end)
    fprintf ('Loading %s%s%s, please wait ...\n', ...
        index.Group {i}, filesep, index.Name {i}) ;
    Problem = ssget (i,index) ;
    save(strcat('matrices_under_2G/',string(c),'.mat'), '-struct', 'Problem');
    c = c+1;
end