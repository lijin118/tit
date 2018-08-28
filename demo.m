clear all
addpath('./libsvm-weights-3.20/matlab');


load('Webcam-decaf-to-dslr-surf.mat');
S=double(S);
S = S ./ repmat(sqrt(sum(S.^2,2)),1,size(S,2));
T=double(T);Ttest=double(Ttest);
T = T ./ repmat(sqrt(sum(T.^2,2)),1,size(T,2));
Ttest = Ttest ./ repmat(sqrt(sum(Ttest.^2,2)),1,size(Ttest,2));



Data.T = T;
Xt = [T;Ttest];
Xs = S;
Xt = Xt ./ repmat(sqrt(sum(Xt.^2,2)),1,size(Xt,2));
Data.T_Label = T_Label;
Ys = S_Label;
Yt = Ttest_Label;

model = svmtrain([],T_Label,T,'-t 0 -q');
[Cls, svm_acc,~] = svmpredict(Yt,Ttest, model, '-q');
fprintf('SVM: acc = %f. \n', svm_acc(1));

% Set algorithm parameters
options.k = 100;             % #subspace bases
options.lambda =1;       % regularizer
options.T = 5;             % #iterations,
options.ker = 'linear';     % kernel type


options.beta=0.1;
options.delta=0.01;

Goptions=[];
Goptions.model='sig';
Goptions.k=8;
Goptions.NeighborMode='KNN';
Goptions.bNormalized=1;
Goptions.WeightMode='Cosine';%'HeatKernel';%'Cosine';
Goptions.t = 0.5;
Goptions.NeighborMode = 'Supervised';
Goptions.gnd = [Ys;T_Label;Cls];
Goptions.weight=[];
%para={0.1,0.2,0.5,1,1.5,2,3,4,5,10};
% for i=1:length(para)
% options.delta =-para{i};
% disp('------------------------------------------------------');
% disp(options.delta);
% disp('------------------------------------------------------');

[Acc,Cls,Z,A,weight] = TIT(Xs',Xt',Ys,Yt,Data.T',Data.T_Label,options,Goptions);
% end

