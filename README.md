# DataMining-Homework
## 1.   Use two visualization techniques
针对给定的数据， 结合其特性选择两个visualize的方法， 并说说从数据中找到了什么规律

## 2. Decision tree
不调用其他库函数，自行实现decision tree
## 3. Naive bayes classifier
自行实现NBC, 要求：给定数据（包含离散变量， 连续变量）， 对连续变量使用两种方式求其条件概率


NBC 问题简述：
根据 bayes rule: 
$$\begin{aligned}
P(Y|X1, X2, X3)=P(X1, X2, X3, Y)/P(X1, X2, X3)
\end{aligned}$$
 
根据chain rule与条件独立性质: 
$$\begin{aligned}
P(X1, X2, X3, Y)&=P(X1, X2, X3|Y)* P(Y)\\
&= P(X1|X2, X3, Y)* P(X2, X3, Y) \\
&= P(X1|Y)* P(X2, X3, Y)\\
&=P(X1|Y)* P(X2|X3, Y) * P(X3, Y) \\
&=P(X1|Y)* P(X2|Y)* P(X3, Y)\\
& = P(X1|Y)* P(X2|Y)* P(X3|Y)  * P(Y)
\end{aligned}$$

最终根据P(Y|X1, X2, X3)的大小， 输出结果（举例：
P(Y=yes|X1, X2, X3)=0.2, P(Y=no|X1, X2, X3)=0.8, then the output result should be "yes"）


## 4. Hidden marcov model 
任务：给定模型以及观测序列o， 应用Viterbi算法求出最可能的状态序列
- 观测序列, 状态序列的解释：
举一个小例子coin tossing， 有两枚硬币， 一个是 fair  coin, 一个是loaded coin. 假设扔硬币的次序是Fair, Fair, Load, Fair这个序列就是状态序列
假设扔完硬币记录正反面的结果是 Head, Head, Tail, Tail,  这个序列就是观测序列