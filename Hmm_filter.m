function alphaNew = Hmm_filter(Obs,Model,alpha)
    if nargin <=2
        error('Need to initialized alpha\n');
    end

    A_T = Model.A';
    for i =1:Model.Num_state
        Temp_E(i,1) = Hmm_Gaussian_Emission(Obs,Model.E{i,1},Model.E{i,2});
    end
    alphaNew = Temp_E.*A_T*alpha;
    alphaNew = alphaNew./sum(alphaNew);
    
end