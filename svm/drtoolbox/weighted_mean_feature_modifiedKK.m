 function [ weigthed_matrix ] = weighted_mean_feature_modifiedKK( labels, img, sup_img)
%=================================================================================
%This function is used to extract spatial information among superpixels
%input arguments:  labels          : superpixel segmentation map
%                  img             : dimension-reduced HSI
%                  sup_img         : superpixels image                  
%output arguments: weigthed_matrix : feature matrix among each superpixels 
%=================================================================================
MaxSegments=max(labels(:));
[no_lines, no_rows, no_bands]=size(img);
s=[no_lines no_rows];
labels=double(labels);
alfa=2;
beta=0.7;
sup_img=sup_img';
for i=0:MaxSegments
    supind=find(labels==i);
    [M,N]=size(supind);
    if M<1
        continue;
    end
    
    mainclass=labels(supind);
    [a,b]=ind2sub(s,supind);    
    centebrseed=[floor(mean(a)),floor(mean(b))]; 
    
    n1=diag(labels(max(1,a-1),b));            % top
    n2=diag(labels(min(no_lines,a+1),b));     % bottom
    n3=diag(labels(a,max(1,b-1)));            % left
    n4=diag(labels(a,min(no_rows,b+1)));      % right
    a=unique([n1;n2;n3;n4]);    
    a(a==i)=[];
    meanv=sup_img(a(:)+1,:);    % +1 because the superpixel counting from 0
%     meanv=mean(meanv);
    centerv=sup_img(i+1,:);     % +1 because the superpixel counting from 0
    [size1,~] = size(meanv);
    w = zeros(size1, 1);
    WA = zeros(no_bands, size1);
    for ii = 1: size1
        w(ii,1) = exp(-((norm(meanv(ii,:) - centerv,2))^2)/5e7);   % here I use 5e7 instead of 500 (as stated in the paper), beacuase the scale doesn't seem correct
%         w(ii,1) = norm(meanv(ii,:) - centerv,2)^2;
        WA(:,ii) = meanv(ii,:);
    end
    w = w./sum(w,1);
    WA = transpose(WA*w);

    
    for j=1:M
        weigthed_matrix(:,supind(j))=WA;
    end
     
end
weigthed_matrix=weigthed_matrix';
weigthed_matrix=reshape(weigthed_matrix,no_lines, no_rows, no_bands);
