function [Model llh]= Hmm_EM(Obs,Model,Max_iter)

    if ~isstruct(Model)
        error('Model is not ready yet.')
    end
 
    [Dim_Obs, Num_Obs] = size(Obs);
    
    
    %EO = Model.E;
    Emission_Prob = zeros(Model.Num_state,Num_Obs);
    
    llh(1) = -inf;
    tol = 1e-5;
    Zitter = 0.01;
    for it = 2:Max_iter+1
        
        A = Model.A;     A_T = A';
        Pi = Model.Pi;
        
        % EM - Expectation
       
        for jj = 1:Model.Num_state
            for ii = 1:Num_Obs
                %fprintf('iteration : %d \n',it);
                Emission_Prob(jj,ii) = Hmm_Gaussian_Emission(Obs(:,ii),Model.E{jj,1},Model.E{jj,2} + Zitter * eye(Dim_Obs));
            end
        end
        
        alpha(:,1) = Pi.*Emission_Prob(:,1);   
        c(1) = SumProb(alpha(:,1));
        alpha(:,1) = alpha(:,1)./c(1);
        
        for ll = 2:Num_Obs
            alpha(:,ll) = Emission_Prob(:,ll).*(A_T*alpha(:,ll-1));
            c(ll) = SumProb(alpha(:,ll));
            alpha(:,ll) = alpha(:,ll)./c(ll);
        end
        
        beta(:,Num_Obs) = ones(Model.Num_state,1);
        for ll = Num_Obs:-1:2
            beta(:,ll-1) = A*(Emission_Prob(:,ll).*beta(:,ll))./c(ll);
        end
        
        gamma = alpha.*beta;
        
        llh(it) = sum(log(c(c>0)));
        if llh(it)-llh(it-1) < tol*llh(it)
            break;
        end
        
        % EM - Maximization
        for ii = 2:Num_Obs
            xi{ii} = A.*(alpha(:,ii-1)*(beta(:,ii).*Emission_Prob(:,ii))'); 
        end
        Anew = sum(cat(3,xi{:}),3);
        Anew = Anew./SumProb(Anew,2);
        
        Pi = gamma(:,1)./SumProb(gamma(:,1));
        
        for i = 1:Model.Num_state
            Mu_new{i,1} = ((gamma(i,:)*Obs')./sum(gamma(i,:)))';
            Sigma_new{i,1} = (gamma(i,:).*(Obs-Mu_new{i,1})*(Obs-Mu_new{i,1})')./sum(gamma(i,:));        
        end
  
        % Update parameter/
        Model.A = Anew;
        Model.Pi = Pi;
        
        for i = 1:Model.Num_state
            Model.E{i,1} = Mu_new{i,1};
            Model.E{i,2} = Sigma_new{i,1}; 
        end
        
    end
    fprintf('Hmm_EM Training done\n');
end