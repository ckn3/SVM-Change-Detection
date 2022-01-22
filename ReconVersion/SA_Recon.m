function recon = SA_Recon(HSI,Y,NumPerClass,NumTrial)
%SA_RECON Summary of this function goes here
%   Detailed explanation goes here

K=length(unique(Y));
num_perclass = repmat(NumPerClass,1,K);

global scale_sp;
scale_sp = [1 2 3 4 5];

[row_max, col_max,~] = size(HSI);
im_gt_1d = reshape(Y,1,row_max*col_max);
ind_map = reshape(1:length(im_gt_1d),[row_max,col_max]);
pca1 = reshape(pca(reshape(HSI,row_max*col_max,[]),1),row_max,col_max,[]);

index = ind_map(:);

gm = 385;
len = shape_adaptive(pca1,gm);

len = reshape(len,row_max*col_max,[])';
len = scale_sp(len);
recon=[];

sigma  = compSigma(HSI,Y,num_perclass,NumTrial,20);

for i=1:length(index)
    row = mod(index(i),row_max);
    if row == 0
        row = row_max;
    end
    col = ceil(index(i)/row_max);
    lens = len(:,index(i));
   
    pixs_xy=PtsSaRi(lens,row,col);

    
    pixs_num = size(pixs_xy,1);
    X_ind = [];
    for j = 1:1:size(pixs_xy,1)
        temp = ind_map(pixs_xy(j,1),pixs_xy(j,2));
        X_ind = [X_ind temp];
    end

    Sneigh=[];
    
    for j=1:pixs_num
        Sneigh=[Sneigh;HSI(pixs_xy(j,1),pixs_xy(j,2),:)];
        if pixs_xy(j,1)==row && pixs_xy(j,2)==col
            k=j;
        end
    end
    Sneigh=reshape(Sneigh,size(Sneigh,1)*size(Sneigh,2),size(Sneigh,3));

    vec = weightVec(Sneigh,k,sigma)';
    R=vec*Sneigh;
    recon=[recon;R];

end

end




function [Sigma,training_pixels,training_labels]=compSigma(data,label,num_perclass,trial_num,pect)
% Input
%    data            :   M*N*B, 3D data
%    label           :   M*N, label
%    num_perclass    :   number of training pixels in each class
%    trial_num       :   10
% Output
%    S               :   computed sigma value
%    training_pixels :   n*B*trial_num, n is total number of training
%                        pixels in each trial
%    training_labels :   n*1, vector
%


[a,b,c]=size(data);
img1=reshape(data,[a*b,c]);
rng('default')
rng(1) %random seed

Train_Label = [];
Train_index = [];
training_pixels=[];

for ii = 1: length(num_perclass)
   index_ii =  find(label == ii);
   class_ii = ones(length(index_ii),1)* ii;
   Train_Label = [Train_Label class_ii'];
   Train_index = [Train_index index_ii'];   
end
trainall = zeros(2,length(Train_index));
trainall(1,:) = Train_index;
trainall(2,:) = Train_Label;

for trial=1:trial_num
    indexes =[];
    
    for i = 1: length(num_perclass)
        W_Class_Index = find(Train_Label == i);
        Random_num = randperm(length(W_Class_Index));
        Random_Index = W_Class_Index(Random_num);
        Tr_Index = Random_Index(1:num_perclass(i));
        indexes = [indexes Tr_Index];
    end
    indexes = indexes';
    train_SL = trainall(:,indexes);
    train_samples = img1(train_SL(1,:),:);
    training_labels= train_SL(2,:)';
    
    training_pixels=cat(3,training_pixels,train_samples);
end



S=zeros(trial_num,1);
S2=zeros(trial_num,1);
for i=1:trial_num
    [s1,s2] = computeSigma(training_pixels(:,:,i),training_labels);
    S(i)=prctile(s1,100-pect);
    S2(i)=prctile(s2,pect);
end
Sigma=mean(S);
end

function [S,S2] = computeSigma(X,Y)

K = length(unique(Y));
S=[];
S2=[];

for i = 1:K
    Xtemp = X(Y==i,:);
    pcr = corrcoef(Xtemp');
    s = unique(pcr);
    S = [S,s'];
    Xtemp = X(Y==i,:);
    Xtemp2 = X(Y~=i,:);
    pcr = corrcoef([Xtemp',Xtemp2']);
    pcr = pcr((size(Xtemp,1)+1):end,1:size(Xtemp,1));
    s = unique(pcr(:));
    S2 = [S2,s'];
end

S=S(S>0);
S2=unique(S2(S2>0));
end

function vec = weightVec(Sneigh,k,Sigma)

pcr = corrcoef(Sneigh');
ind = (pcr(:,k)>Sigma);
if sum(ind)==1
    vec = ind;
else
    D = pdist2(Sneigh,Sneigh(k,:));
    Ds = D(ind);
    sigma1 = prctile(Ds,50);
    gk = exp(-(D.^2)./(sigma1^2));
    vec = zeros(length(gk),1);
    vec(ind) = gk(ind);
    vec = vec./sum(vec);
end

if sum(isnan(vec))>0
    vec = zeros(size(vec));
    vec(k) = 1;
end

end
