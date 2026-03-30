%%%%%%%%%%%%%%%%%%% 线性不可分SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % 样本量大小
center1 = [1,1];        % 第一类数据中心
center2 = [3,3];        % 第二类数据中心
X = zeros(2*n,2);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = zeros(2*n,1);       % 类别标签
X(1:n,:) = ones(n,1)*center1 + randn(n,2);
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % 第一类数据标签为1，第二类为-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%%  SVM模型   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  学生实现,求出SVM的参数(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(2,1);
b = zeros(1);               % SVM: y = x*w + b

%%%%%%%% 使用线性增广拉格朗日法训练模型
K = 5000000;
alpha = zeros(2*n,1);     
lambda = zeros(1,1);    % 对偶变量
beta = 1;
C = 1;
eta = 0.0001;

hatX = (Y*ones(1,2)) .*X;
hatXX = hatX*hatX';
for k=1:K
   
   hat_alpha = alpha - eta * ( hatXX*alpha - ones(2*n,1) + lambda*Y + beta*Y'*alpha*Y ); 
   alpha = hat_alpha;
   alpha(alpha<0) = 0;
   alpha(alpha>C) = C;
  
   lambda = lambda + beta* ( Y'*alpha );
   
   func(k) = alpha'*hatXX*alpha/2 - sum(alpha);    % 目标函数
   cons(k) = abs(Y'*alpha);                           % 约束
end

w = hatX'*alpha;
ttt = 1;
for j=1:2*n
   if 0 < alpha(j) && alpha(j) < C
       b = Y(j) - sum( (alpha*ones(1,2)) .*hatX ) * X(j,:)';
       ttt = 0;
       b
       %break;
   end
end
if ttt==1
   disp('wrong') 
end

figure(2)                                          % 画出训练过程中目标函数下降情况，方便观察算法是否收敛；将eta调小时，算法不收敛
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot( func,'r','LineWidth',1,'MarkerSize',10); 
hold on;
xlabel('number of iterations');
ylabel('objective value');

figure(3)                                          % 画出训练过程中约束满足情况，方便观察算法是否收敛
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot( cons,'r','LineWidth',1,'MarkerSize',10); 
hold on;
xlabel('number of iterations');
ylabel('objective value');

%%%%%%%%%%%%%%%%  分类器可视图  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% 即画出 x*w + b =0 的图像 %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % 分类界面
                                                           % x1为分类界面横轴，y1为纵轴
y2 = ( ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);
y3 = ( -ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);  %画出间隔边界

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % 画分类界面
hold on;
plot( x1,y2,'k-.','LineWidth',1,'MarkerSize',10);                         % 画分间隔边界
hold on;
plot( x1,y3,'k-.','LineWidth',1,'MarkerSize',10);                         % 画分间隔边界
hold on;
plot(X(alpha>0,1),X(alpha>0,2),'rs','LineWidth',1,'MarkerSize',10);    % 画支持向量
hold on;
plot(X(alpha<C&alpha>0,1),X(alpha<C&alpha>0,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);    % 画间隔边界上的支持向量
hold on;
xlabel('x axis');
ylabel('y axis');
set(gca,'Fontsize',10)
legend('class 1','class 2','classification surface','boundary','boundary','support vectors','support vectors on boundary');

