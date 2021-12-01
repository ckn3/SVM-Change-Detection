function [class_label_denoised, besta1a2] = denoisingL1L2(predict_label_prob, par,par2,train_map, test_SL, K_Known)
OA = 0;AA=0;kappa=0;

for a1=par
    for a2 = par2

        denoise_predict_tensor = l2_l1_aniso_l2_less_ADMM_2dir(predict_label_prob,a1,a2,train_map==0,5);

        [~,class_label_denoised_temp] = max(denoise_predict_tensor,[],3); %% classification rule in stage 2
        class_label_denoised_temp = reshape(class_label_denoised_temp,[],1);
        [OA_temp,kappa_temp,AA_temp,~] = calcError(test_SL(2,:)-1,class_label_denoised_temp(test_SL(1,:))'-1,1:K_Known);
        
        if (OA_temp+kappa_temp)>(OA+kappa)
            OA = OA_temp;AA=AA_temp;kappa=kappa_temp;
            besta1=a1;besta2=a2;
            class_label_denoised = class_label_denoised_temp;
        end
        
    end
end
besta1a2 = [besta1,besta2];



end