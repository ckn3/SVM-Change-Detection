function [Jac,F1] = JacF1(gt, pred, K_Known)

gt = gt(:);
pred = pred(:);
Jac = jaccard(gt,pred);
C = confusionmat(gt,pred);

F1 = zeros(K_Known,1);
Cc = sum(C);
Cr = sum(C');

for i = 1:K_Known
    F1(i) = 2.*C(i,i)./(Cc(i)+Cr(i));
end

end