%%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % 样本量大小
X = rand(n,2)*10;        % n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = zeros(n,1);          % 类别标签

for i=1:n
   if 0<X(i,1) && X(i,1)<3 && 0<X(i,2) && X(i,2)<3              % 根据x和y轴坐标确定分类      
       Y(i) = 1;
   end
   if 0<X(i,1) && X(i,1)<3 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 2;
   end
   if 0<X(i,1) && X(i,1)<3 && 7<X(i,2) && X(i,2)<10
       Y(i) = 3;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 0<X(i,2) && X(i,2)<3
       Y(i) = 4;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 5;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 7<X(i,2) && X(i,2)<10
       Y(i) = 6;
   end
   if 7<X(i,1) && X(i,1)<10 && 0<X(i,2) && X(i,2)<3
       Y(i) = 7;
   end
   if 7<X(i,1) && X(i,1)<10 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 8;
   end;
   if 7<X(i,1) && X(i,1)<10 && 7<X(i,2) && X(i,2)<10
       Y(i) = 9;
   end;
end
X = X(Y>0,:);                                                    % 注意X是在[0,10]*[0,10]范围内均匀生成的，而我们只标出了一部分X，类别之间的白色间隔中的点没有标，因此需要将这些点去掉
Y = Y(Y>0,:);                                                    % X(Y>0,:)表示只取X中对应的Y大于0的行，这是因为白色间隔中的点的Y都为0
n = length(Y);                                                   % 去除掉白色间隔剩下的点

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点    X(Y==1,1)表示类别为1（Y==1）的点的第一维度坐标，X(Y==1,2)表示类别为1的点的第二维度坐标
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % 画第二类数据点
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % 画第三类数据点
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % 画第四类数据点
hold on;
plot(X(Y==5,1),X(Y==5,2),'m*','LineWidth',1,'MarkerSize',10);            % 画第五类数据点
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % 画第六类数据点
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % 画第七类数据点
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % 画第八类数据点
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % 画第九类数据点
hold on;
xlabel('x axis');
ylabel('y axis');

%%%%%%%%%%%%%%%%%%%  测试  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% 生成测试数据:与训练数据同分布 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 500;                % 测试样本量大小
Xt = rand(m,2)*10;       
Yt = zeros(m,1);
for i=1:m
   if 0<Xt(i,1) && Xt(i,1)<3 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 1;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 2;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 3;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 4;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 5;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 6;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 7;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 8;
   end;
   if 7<Xt(i,1) && Xt(i,1)<10 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 9;
   end;
end
Xt = Xt(Yt>0,:);
Yt = Yt(Yt>0,:);
m = length(Yt);

figure(2)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % 画第二类数据点
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % 画第三类数据点
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % 画第四类数据点
hold on;
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);            % 画第五类数据点
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % 画第六类数据点
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % 画第七类数据点
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % 画第八类数据点
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % 画第九类数据点
hold on;
plot(Xt(:,1),Xt(:,2),'ms','MarkerFaceColor','m','LineWidth',1,'MarkerSize',10);            % 画测试数据点   Xt(:,2)表示Xt的第二列，包括所有行
hold on;
xlabel('x axis');
ylabel('y axis');
%%%%%%%%%%%%%%%%%%  K-近邻算法：学生实现     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  给出模型的预测输出，并与测试数据的真实输出比较，计算错误率     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = 10;
Ym = zeros(m,1);
for i=1:m
   dist = zeros(1,n);
   for j=1:n
       dist(j) = (X(j,1)-Xt(i,1))^2 + (X(j,2)-Xt(i,2))^2;            % 计算i和j距离
%       dist(j) = abs(X(j,1)-Xt(i,1)) + abs(X(j,2)-Xt(i,2));         % 不同的距离度量
%       dist(j) = max( [ abs(X(j,1)-Xt(i,1)) , abs(X(j,2)-Xt(i,2)) ] );            % 不同的距离度量
   end
   [dists,index] = sort(dist);                                       % 对距离排序，dists是从小到大的距离，index是相应元素在原始向量dist中的位置 
   Ys = Y(index(1:K));                                               % 只取距离最小的K个点
   count = tabulate(Ys);                                             % 对这个K个点的类别计数，count第一列表示这K个点都属于那些类，第二列表示每个类出现次数
   [maxc,maxj] = max(count(:,2));                                    % 取出现次数最多的类
   Ym(i) = count(maxj,1);                                            % 将出现次数最多的类作为点i的类
end

[Yt';Ym']                                         % 第一行为测试样本真实输出，第二行为模型输出
sum(Yt~=Ym)                                       % 计数预测错误几个数据点                     
sum(Yt~=Ym)/m                                     % 计算错误率

figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % 画第二类数据点
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % 画第三类数据点
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % 画第四类数据点
hold on;
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);            % 画第五类数据点
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % 画第六类数据点
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % 画第七类数据点
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % 画第八类数据点
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % 画第九类数据点
hold on;
plot(Xt(Ym==1,1),Xt(Ym==1,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(Xt(Ym==2,1),Xt(Ym==2,2),'ks','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);            % 画第二类数据点
hold on;
plot(Xt(Ym==3,1),Xt(Ym==3,2),'bs','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % 画第三类数据点
hold on;
plot(Xt(Ym==4,1),Xt(Ym==4,2),'gs','MarkerFaceColor','g','LineWidth',1,'MarkerSize',10);            % 画第四类数据点
hold on;
plot(Xt(Ym==5,1),Xt(Ym==5,2),'bs','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % 画第五类数据点
hold on;
plot(Xt(Ym==6,1),Xt(Ym==6,2),'cs','MarkerFaceColor','c','LineWidth',1,'MarkerSize',10);            % 画第六类数据点
hold on;
plot(Xt(Ym==7,1),Xt(Ym==7,2),'bs','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % 画第七类数据点
hold on;
plot(Xt(Ym==8,1),Xt(Ym==8,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);            % 画第八类数据点
hold on;
plot(Xt(Ym==9,1),Xt(Ym==9,2),'ks','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);            % 画第九类数据点
hold on;
xlabel('x axis');
ylabel('y axis');

