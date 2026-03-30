%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % 样本量大小
center1 = [1,1];        % 第一类数据中心
center2 = [6,6];        % 第二类数据中心
%线性可分数据：center2 = [6,6]；线性不可分数据，改为center2 = [3,3]
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

C = 1;           % 惩罚参数C
tol = 0.001;    % 容忍度
max_passes = 50;  % 最大迭代次数

% 初始化变量
alpha = zeros(2*n,1);
b = 0;
passes = 0;

% 计算Gram矩阵
K = X * X';

% SMO算法主循环
while passes < max_passes
    num_changed_alphas = 0;
    for i = 1:2*n
        % 计算误差项
        E_i = sum(alpha.*Y.*(K(:,i))) + b - Y(i);
        
        % 如果误差项超出容忍度并且alpha在(0, C)之间，选择第二个变量j
        if (Y(i)*E_i < -tol && alpha(i) < C) || (Y(i)*E_i > tol && alpha(i) > 0)
            % 选择违反KKT条件最严重的alpha作为第一个优化变量alpha_i
            j = randi([1 2*n],1);  % 随机选择一个j
            while j == i
                j = randi([1 2*n],1);
            end
            
            % 计算两个变量的误差项
            E_j = sum(alpha.*Y.*(K(:,j))) + b - Y(j);
            
            % 保存旧的alpha值
            alpha_i_old = alpha(i);
            alpha_j_old = alpha(j);
            
            % 计算alpha的上下界
            if Y(i) ~= Y(j)
                L = max(0, alpha(j) - alpha(i));
                H = min(C, C + alpha(j) - alpha(i));
            else
                L = max(0, alpha(i) + alpha(j) - C);
                H = min(C, alpha(i) + alpha(j));
            end
            
            % 如果上下界相等，则跳过这个迭代步骤
            if L == H
                continue;
            end
            
            % 计算eta
            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            
            if eta >= 0
                continue;
            end
            
            % 更新alpha_j
            alpha(j) = alpha_j_old - Y(j)*(E_i - E_j) / eta;
            
            % 修剪alpha_j
            alpha(j) = max(L, alpha(j));
            alpha(j) = min(H, alpha(j));
            
            % 如果alpha_j的变化量很小，则跳过这个迭代步骤
            if abs(alpha(j) - alpha_j_old) < tol
                continue;
            end
            
            % 更新alpha_i
            alpha(i) = alpha_i_old + Y(i)*Y(j)*(alpha_j_old - alpha(j));
            
            % 更新b
            b_i = b - E_i - Y(i)*(alpha(i) - alpha_i_old)*K(i,i)...
                - Y(j)*(alpha(j) - alpha_j_old)*K(i,j);
            b_j = b - E_j - Y(i)*(alpha(i) - alpha_i_old)*K(i,j)...
                - Y(j)*(alpha(j) - alpha_j_old)*K(j,j);
            
            if alpha(i) > 0 && alpha(i) < C
                b = b_i;
            elseif alpha(j) > 0 && alpha(j) < C
                b = b_j;
            else
                b = (b_i + b_j) / 2;
            end
            
            % 更新误差缓存E
            E = sum(alpha.*Y.*(K(:,i))) + b - Y';
            
            num_changed_alphas = num_changed_alphas + 1;
        end
    end
    
    % 检查是否有alpha的变化量很小，如果有，则增加迭代次数，直到达到最大迭代次数
    if num_changed_alphas <= 0
        passes = passes + 1;
    else
        passes = 0;
    end
end

% 寻找支持向量
support_indices = find(alpha > 1e-4);
support_vectors = X(support_indices,:);
support_labels = Y(support_indices);
support_alpha = alpha(support_indices);

% 计算权重向量w
w = (support_alpha .* support_labels)' * support_vectors;

% 计算间隔边界上的支持向量
margin_indices = find(alpha > 0 & alpha < C);
margin_vectors = X(margin_indices,:);
margin_labels = Y(margin_indices);

% 计算偏置b
b = mean(margin_labels - margin_vectors * w');
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
