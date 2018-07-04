function [Acc,Cls,Z,A,weight] = TIT(Xs,Xt,Ys,Yt,T_data,T_label,options,Goptions)

if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'lambda')
    options.lambda = 1.0;
end
if ~isfield(options,'T')
    options.T = 10;
end
if ~isfield(options,'ker')
    options.ker = 'linear';
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'beta')
    options.beta = 1.0;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
lambda = options.lambda;
T = options.T;
ker = options.ker;
gamma = options.gamma;
data = options.data;
beta=options.beta;

% Set predefined variables
%
[ds,ns] = size(Xs);
[dt,nt] = size(Xt);
[dst,nst]=size(T_data);
if(ds==dt)
    X = [Xs,Xt];
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
else
    X = [Xs,zeros(ds,nt);zeros(dt,ns),Xt];
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
end
n = ns+nt;

% Construct kernel matrix
K = kernel(ker,X,[],gamma);


% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);

% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e';
M = M/norm(M,'fro');

%Construct Graph
if strcmp(Goptions.model,'sig')
    %Goptions.gnd=[Ys;Yt0];
    W= constructW(X',Goptions);
    D = diag(full(sum(W,2)));
    W1 = -W;
    for i=1:size(W1,1)
        W1(i,i) = W1(i,i) + D(i,i);
    end
else
    [Ww,Wb]=ConG(Goptions.gnd_train,Goptions,X','srge');
    W1=Ww-options.gamma*Wb;
end


G = speye(n);
R=options.delta*eye(n);
R(1:ns,1:ns)=0;
Acc = [];
weight=zeros(ns+nst,1);
for t = 1:T
    [A,~] = eigs(K*(M+beta*W1-R)*K'+lambda*G,K*H*K',k,'SM');
    if(ds==dt)
    G(1:ns,1:ns) = diag(sparse(1./(sqrt(sum(A(1:ns,:).^2,2)+eps))));
    else
    G = diag(sparse(1./(sqrt(sum(A.^2,2)+eps))));
    end
    Z = A'*K;
   

    %model = svmtrain(Ys,Z(:,1:ns)','-t 0 -q');
    loss=trace(A'*K*(M+beta*W1-R)*K'*A)+sum(sqrt(sum(A(1:ns,:).^2,2)));
    model = svmtrain(Goptions.weight,[Ys;T_label],Z(:,1:ns+nst)','-t 0 -q');
    [Cls, svm_acc,~] = svmpredict(Yt,Z(:,ns+nst+1:n)', model, '-q');
    fprintf('SVM: acc = %f. , loss= %f\n', svm_acc(1),loss);
    
    %Goptions.NeighborMode = 'Supervised';
    Goptions.gnd = [Ys;T_label;Cls];
    
    if strcmp(Goptions.model,'sig')
        W= constructW(X',Goptions);
        D = diag(full(sum(W,2)));
        W1 = -W;
        for i=1:size(W1,1)
            W1(i,i) = W1(i,i) + D(i,i);
        end
    else
        [Ww,Wb]=ConG(Goptions.gnd_train,Goptions,X','srge');
        W1=Ww-options.gamma*Wb;
    end
    
    for c=1:length(unique(Cls))
        idt= Cls==c;
        %ids= Ys==c;
        Zs=Z(:,1:ns)';
        Zt=Z(:,ns+nst+1:n)';
        %Ssamps=Zs(ids,:);
        Tsamps=Zt(idt,:);
        [idx, dist] = knnsearch(Zs,Tsamps,'dist','cosine','k',10);
        idx=idx(:);
        for i=1:length(idx)
           weight(idx(i,:),:)=weight(idx(i,:),:)+1;
        end
    end
     
end


end
