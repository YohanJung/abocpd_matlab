clear;
useReal = true;

if useReal

    load data/30_industry.mat;

    Tlearn = 5500; %corresponds roughly to 1985
    % get the time stamp out of the matrix
    year = floor(thirty_industry(:, 1) / 10000);
    month = mod(floor(thirty_industry(:, 1) / 100), 100);
    day = mod(thirty_industry(:, 1), 100);
    time = datenum([year month day]);
    % TODO only fit the covariance and mean from the training part ONLY
%     X = whitten(thirty_industry(:, 2:end), Tlearn);
%     df = 8;
%     X = tcdf(X, df);
%     X = norminv(X);
    
    X = thirty_industry(:, 2:end);
    X = log((X./100) + 1);
    X = X.*100;
    Xtrain = X(1:Tlearn,:);
    Xtest = X(Tlearn+1:end,:);
    
    %whitening code: seems like a bad idea!
%     [U,L] = eig(cov(Xtrain));
%     eigvals = diag(L);
%     Xtrain = (diag(1./sqrt(eigvals))) * U' * (bsxfun(@minus, Xtrain, mean(Xtrain)))';
%     Xtrain = Xtrain';
    
    Xtrain = bsxfun(@minus, Xtrain, mean(Xtrain));
    theta_init = [0.001; 5; 35];
    theta_init'
    [hazard_params, model_params, nlml_train, R_train, S_train] = learnFC(Xtrain, theta_init);
    %[theta, nlml] = minimize(theta_init, @bocpd_deriv_sparse_wrap, -20, Xtrain, 'FCsimple', 'constant_h', 1);
    [h1, h2] = plotS(S_train, Xtrain, [], time(1:Tlearn)); datetick(h1); datetick(h2);

%     Xtest = (diag(1./sqrt(eigvals))) * U' * (bsxfun(@minus, Xtest, mean(Xtrain)))'; %or mean(Xtest)?
%     Xtest = Xtest';
    Xtest = bsxfun(@minus, Xtest, mean(Xtrain));
    [R_test S_test nlml_test] = bocpd_sparse(hazard_params', model_params', Xtest, 'constant_h', 'FCsimple', 0.001);
    [h1, h2] = plotS(S_test, Xtest, [], time(Tlearn+1:end)); datetick(h1); datetick(h2);
    
else

    T = 500;
    theta_init = [10/T; 5; 35];
    [X changePoints] = FCrnd(T, 30, theta_init(2), theta_init(3), theta_init(1));
    X = X';
    [hazard_params, model_params, nlml_train] = learnFC(X, theta_init*2);
    [R S nlml] = bocpd_sparse(hazard_params', model_params', X, 'constant_h', 'FCsimple', 0.001);
    plotS(S, X, changePoints);
    
end
    