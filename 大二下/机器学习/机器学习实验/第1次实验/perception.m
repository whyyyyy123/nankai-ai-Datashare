clear all;
%%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % 样本量大小
center1 = [1,1];        % 第一类数据中心
center2 = [3,4];        % 第二类数据中心
X = zeros(2*n,2);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = zeros(2*n,1);       % 类别标签
X(1:n,:) = ones(n,1)*center1 + randn(n,2);           %生成数据：中心点+高斯噪声
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % 第一类数据标签为1，第二类为-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%%  感知机模型   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  学生实现,求出感知机模型的参数(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(2,1);
b = zeros(1);               % 感知机模型 y = x*w + b

%%%%%%%% 使用梯度下降法训练模型；即最小化f(w,b) = -sum_M( ( X*w + ones(2*n,1)*b ).* Y )
K = 1000;
eta1 = 0.01;
eta2 = 0.01;
for k=1:K
    pred = ( X*w + ones(2*n,1)*b ).* Y;
    all_correct = 1;
    for i=1:2*n
        if pred(i)<=0
            w = w + eta1 * X(i,:)' * Y(i);                
            b = b + eta2 * Y(i); 
            all_correct = 0;
        end
    end
    tt = X*w + ones(2*n,1)*b;
    tt'
    
    if all_correct == 1
        break;
    end
end

%%%%%%%%%%%%%%%%  分类器可视图  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% 即画出 x*w + b =0 的图像 %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;                                     % 首先采样若干x轴点，然后计算分类面上对应的y轴点，有了x轴和y轴点，即可画出图像
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % 分类界面,length()表示向量长度
                                                           % x1为分类界面横轴，y1为纵轴
figure(2)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                        % 画分类界面
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2','classification surface');

%%%%%%%%%%%%%%%%%%%  测试  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% 生成测试数据:与训练数据同分布 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 10;                  % 测试样本量大小
Xt = zeros(2*m,2);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Yt = zeros(2*m,1);       % 类别标签
Xt(1:m,:) = ones(m,1)*center1 + randn(m,2);
Xt(m+1:2*m,:) = ones(m,1)*center2 + randn(m,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Yt(1:m) = 1; 
Yt(m+1:2*m) = -1;        % 第一类数据标签为1，第二类为-1 

figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);              % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);      % 画第二类数据点
hold on;
plot(Xt(1:m,1),Xt(1:m,2),'go','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(Xt(m+1:2*m,1),Xt(m+1:2*m,2),'g*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                        % 画分类界面
hold on;
xlabel('x axis');
xlabel('x axis');
ylabel('y axis');
set(gca,'Fontsize',8);
legend('class 1: train','class 2: train','class 1: test','class 2: test','classification surface');

%%%%%%%%%%%%%%%%%%  学生实现     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  给出模型的预测输出，并与测试数据的真实输出比较，计算错误率     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ym= Xt*w + ones(2*m,1)*b;                         % 模型在测试样本上的输出
Ym(Ym>0) = 1;                                     % Ym>0的数据点的标签为1
Ym(Ym<0) = -1;                                    % Ym<0的数据点的标签为-1
[Yt';Ym']                                         % 第一行为测试样本真实输出，第二行为模型输出
sum(Yt~=Ym)                                       % 计数预测错误几个数据点，sum(Yt~=Ym)计数Yt和Ym中不相等的元素个数                    
sum(Yt~=Ym)/m/2                                   % 计算错误率，总共2m个测试样本