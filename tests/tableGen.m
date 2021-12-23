t = 3;
table = [OAs(t,:)' kappas(t,:)' Jacs(t,:)' mean(F1s(t,:,:),3)'];
tableF1 = reshape(F1s(t,:,:),7,size(F1s,3));