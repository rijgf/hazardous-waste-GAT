# 基于 PPO-Transformer 的多周期危化废弃物收运方案改进算法

> 正文草稿。本文描述的是仓库当前可执行的 PPO-Transformer 基线：算法不从空白状态逐点生成路线，而是以一个完整可行的启发式解为起点，通过八类邻域算子持续改进多周期收运方案。叙述顺序参考向传凯等关于“特征加工—Transformer 编码—决策输出—强化学习训练”的写法，但网络输出、约束处理和训练算法均以本项目实现为准。

## 1 问题求解思路与马尔可夫决策过程

设产废节点集合、危化废弃物类型集合、处理处置节点集合、车辆集合和周期集合分别为 $I$、$S$、$J$、$K$ 和 $T$，由产废节点 $i$ 与危化废弃物类型 $s$ 唯一确定的收运任务节点记为 $n=(i,s)\in N_G$。一个完整收运方案可写为

$$
\mathcal P=\{R_{kt}\mid k\in K,t\in T\},\qquad
R_{kt}=(j,n_1,n_2,\ldots,n_q,j),
$$

其中，$R_{kt}$ 表示车辆 $k$ 在周期 $t$ 的闭合路线，路线从处理处置节点 $j$ 出发并返回该节点。与从空白状态依次选择访问节点的构造式强化学习不同，本文采用改进式强化学习：首先生成完整可行的初始方案 $\mathcal P_0$，然后由智能体在每一步选择“邻域算子—操作对象”组合，将当前方案 $\mathcal P_l$ 转移为候选方案 $\mathcal P_{l+1}$。

在训练迭代开始时，启发式算法将所有产废节点收运任务暂置于末周期，优先选择累计待收量较大的节点，并在车辆容量、危化废弃物兼容性、处理技术匹配和弧可达约束下构造路线。固定随机种子仅用于候选并列时的确定性打破，从而使初始方案可以复现。强化学习状态定义为

$$
s_l=\left(\mathcal P_l,\boldsymbol\omega,\eta_l,\xi_l,\Theta\right),
$$

其中，$\boldsymbol\omega=(\omega_c,\omega_r)^\mathsf T$ 为成本—风险偏好，且 $\omega_c+\omega_r=1$；$\eta_l=l/L$ 为当前步进比例；$\xi_l$ 为连续未改进步数占总步数的比例；$\Theta$ 表示距离、产生量、容量、处理能力、事故概率、风险系数等实例参数。状态并不直接以字典或路线字符串的形式输入网络，而是先转换为固定列宽的异构 token 矩阵。

## 2 初始解与参数信息的 token 化表示

### 2.1 从路线方案恢复完整状态量

给定当前方案 $\mathcal P_l$，算法首先恢复弧选择、收运量、产废节点期初/期末库存、处理处置节点接收量、处理量和处理处置节点期末库存等变量。以产废节点库存为例，令 $BG_{nst}$、$q_{nkt}$ 和 $IG_{nt}$ 分别表示节点 $n$ 在周期 $t$ 的可收运量、车辆收运量和期末库存，则

$$
BG_{nt}=IG_{n,t-1}+g_{nt},\qquad
IG_{nt}=BG_{nt}-\sum_{k\in K}q_{nkt}.
$$

处理处置节点侧令 $R_{jst}$、$BD_{jst}$、$p_{jst}$ 和 $ID_{jst}$ 分别表示接收量、处理前库存、处理量和期末库存，则

$$
BD_{jst}=ID_{js,t-1}+R_{jst},\qquad
p_{jst}=\min\{BD_{jst},\bar p_{jst}a_{js}\},\qquad
ID_{jst}=BD_{jst}-p_{jst},
$$

其中，$a_{js}\in\{0,1\}$ 表示处理处置节点 $j$ 是否具备处理危化废弃物 $s$ 的技术，$\bar p_{jst}$ 为处理能力。由此，初始路线与实例参数被共同展开为能反映“周期—车辆—产废节点—处理处置节点—弧”关系的状态信息。

### 2.2 异构 token 的统一矩阵

> **写作安排说明（供后续定稿参考，不作为正文内容）：** 正文建议只抽象说明七类 token 的设置依据及其分别表征的状态含义，并给出异构 token 组成统一输入矩阵的总体形式；各类 token 的逐维特征公式、字段顺序、缩放方式及中文释义移至附录展开。当前草稿暂时保留下方详细表格，作为正文压缩和附录编写的素材，不表示定稿时需要将全部字段放入正文。

所有 token 均使用 18 维行向量。定义统一封装函数

$$
\phi_\tau(\mathbf z)=
\begin{bmatrix}
\tau/10 & \mathbf z^\mathsf T & \mathbf 0^\mathsf T
\end{bmatrix}^\mathsf T\in\mathbb R^{18},
$$

其中，$\tau\in\{0,1,\ldots,6\}$ 为 token 类型编号，$\mathbf z$ 为该类 token 的有效特征，其余位置补零。不同类别的特征语义如下。

| 类型 $\tau$ | token | 有效特征向量 $\mathbf z$ |
|---:|---|---|
| 0 | 全局方案 | $[\eta_l,\xi_l,C_l/1000,R_l/100,J_l/1000]$ |
|  | 中文含义 | [当前训练进度，连续未改进比例，固定尺度缩放后的总成本、总风险、加权目标值] |
| 1 | 目标偏好 | $[\omega_c,\omega_r]$ |
|  | 中文含义 | [成本偏好权重，风险偏好权重] |
| 2 | 周期 | $[t/t_{\max},G_t/s_G,Q_t/s_G,IG_t/s_{IG},ID_t/s_{ID},P_t/s_P,U_t/\lvert K\rvert]$ |
|  | 中文含义 | [归一化周期编号，当期危化废弃物产生量，当期危化废弃物收运量，产废节点期末库存，处理处置节点期末库存，当期处理量，车辆启用比例] |
| 3 | 车辆—周期 | $[\hat k,\hat t,u_{kt},D_{kt}/s_D,Q_{kt}/Q_k,m_{kt}/\lvert N_G\rvert,A_{kt}/s_A]$ |
|  | 中文含义 | [车辆编号，周期编号，车辆是否启用，路线总长度，车辆载荷率，路线任务占比，路线弧事故概率水平] |
| 4 | 产废节点—危化废弃物—周期 | $[\hat i,\hat s,\hat t,y_{nt},g_{nt}/g_{\max},BG_{nt}/s_Q,Q_{nt}/s_Q,IG_{nt}/s_Q,c_s/c_{\max},r^G_{is}/r^G_{\max}]$ |
|  | 中文含义 | [产废节点编号，危化废弃物类型编号，周期编号，当期是否服务，当期产生量，收运前可用量，实际收运量，收运后库存，危化废弃物危害后果系数，产废节点库存风险系数] |
| 5 | 处理处置节点—危化废弃物—周期 | $[\hat j,\hat s,\hat t,a_{js},\bar p_{jst}/s_Q,R_{jst}/s_Q,BD_{jst}/s_Q,p_{jst}/\max(\bar p_{jst},1),ID_{jst}/s_Q,c^P_{js}/c^P_{\max},r^D_{js}/r^D_{\max}]$ |
|  | 中文含义 | [处理处置节点编号，危化废弃物类型编号，周期编号，处理技术是否匹配，处理能力，处理处置节点接收量，处理前库存，处理能力利用率，处理后库存，单位处理成本，处理处置节点库存风险系数] |
| 6 | 路线弧 | $[\hat k,\hat t,d_{ab}/d_{\max},p_{ab}/p_{\max},q^{\mathrm{cum}}_{ab}/Q_k,h^{\mathrm{type}}_{ab}/\lvert S\rvert]$ |
|  | 中文含义 | [车辆编号，周期编号，弧距离水平，弧事故概率水平，车辆行至该弧时的累计载荷率，已装载危化废弃物种类占比] |

其中，$C_l$ 为当前方案的总成本，即车辆固定成本、行驶成本和处理成本之和；$R_l$ 为当前方案的总风险，即运输风险、共载附加风险、产废节点库存风险和处理处置节点库存风险之和；$J_l$ 为当前成本—风险偏好下的原始加权目标值：

$$
J_l=\omega_c C_l+\omega_r R_l.
$$

$C_l/1000$、$R_l/100$ 和 $J_l/1000$ 仅表示用预设常数进行固定尺度缩放，目的是压缩数值量级、改善网络输入的尺度，并不属于基于最大值、最小值或均值方差的严格归一化，也不能保证结果落入 $[0,1]$。

表中，带“帽”的变量表示集合内顺序编号缩放结果；$G_t$、$Q_t$、$IG_t$、$ID_t$、$P_t$ 和 $U_t$ 分别为周期总产生量、收运量、产废节点库存、处理处置节点库存、处理量和启用车辆数；$u_{kt}$ 表示车辆是否启用；$D_{kt}$、$Q_{kt}$、$m_{kt}$ 和 $A_{kt}$ 分别为路线长度、载荷、任务数和弧事故概率之和；$y_{nt}$ 表示节点是否在该周期被服务；$q^{\mathrm{cum}}_{ab}$ 与 $h^{\mathrm{type}}_{ab}$ 表示车辆行至弧 $(a,b)$ 时的累计载荷和已装载危化废弃物种类数。$s_G,s_{IG},s_{ID},s_P,s_D,s_A,s_Q$ 均由实例最大产生量、容量、最大距离或最大事故概率构造，目的是削弱不同量纲对梯度的影响。

在当前配置中，偏好 token、处理处置节点 token 和路线弧 token 均启用。将所有真实 token 按“全局—偏好—周期—车辆周期—产废节点危化废弃物周期—处理处置节点危化废弃物周期—路线弧”的顺序堆叠，可得

$$
\mathbf X_l=
\begin{bmatrix}
\phi_0^\mathsf T\\
\phi_1^\mathsf T\\
\boldsymbol\Phi_2\\
\boldsymbol\Phi_3\\
\boldsymbol\Phi_4\\
\boldsymbol\Phi_5\\
\boldsymbol\Phi_6
\end{bmatrix}
\in\mathbb R^{M_l\times18}.
$$

真实 token 数量为

$$
M_l=1+1+\lvert T\rvert+\lvert K\rvert\lvert T\rvert
+\lvert N_G\rvert\lvert T\rvert+\lvert J\rvert\lvert S\rvert\lvert T\rvert+
\sum_{k\in K,t\in T}\lvert E(R_{kt})\rvert.
$$

为了支持批训练，矩阵被补零至 $M_{\max}=896$ 行，得到 $\bar{\mathbf X}_l\in\mathbb R^{896\times18}$。若真实 token 数超过上限，程序直接报错而不截断，以避免悄然丢失路线或库存信息。

## 3 Transformer Actor-Critic 网络

### 3.1 分类型线性嵌入

七类 token 的有效字段含义不同，因此网络不共用同一个输入投影，而是为每一类设置独立参数。对第 $i$ 个 token，令其类别为 $\tau_i$，则

$$
\mathbf e_i^{(0)}=\mathbf W_{\tau_i}^{E}\bar{\mathbf x}_i+\mathbf b_{\tau_i}^{E},
\qquad
\mathbf W_{\tau_i}^{E}\in\mathbb R^{d\times18}.
$$

将各行堆叠得到初始嵌入矩阵 $\mathbf E^{(0)}\in\mathbb R^{M_{\max}\times d}$。当前配置取 $d=48$。该处理方式与直接把所有异构特征送入同一线性层相比，能够让网络分别学习全局指标、库存状态、车辆状态和弧状态的尺度及组合方式。当前实现没有额外加入位置编码，token 间的关系主要由类型、实体编号、周期编号及自注意力共同刻画。

### 3.2 Transformer 编码器

对第 $c$ 层、第 $h$ 个注意力头，查询、键和值矩阵为

$$
\mathbf Q_h=\mathbf E^{(c-1)}\mathbf W_h^Q,\qquad
\mathbf K_h=\mathbf E^{(c-1)}\mathbf W_h^K,\qquad
\mathbf V_h=\mathbf E^{(c-1)}\mathbf W_h^V.
$$

缩放点积注意力写为

$$
\operatorname{Attn}_h(\mathbf Q_h,\mathbf K_h,\mathbf V_h)
=\operatorname{softmax}\!\left(
\frac{\mathbf Q_h\mathbf K_h^\mathsf T}{\sqrt{d_h}}+\mathbf B
\right)\mathbf V_h,
$$

其中，$d_h=d/H$，$H=4$ 为注意力头数，$\mathbf B$ 为 padding mask 对应的加性矩阵。各注意力头拼接并线性映射：

$$
\operatorname{MHA}(\mathbf E)=
\operatorname{Concat}(\operatorname{Attn}_1,\ldots,\operatorname{Attn}_H)\mathbf W^O.
$$

随后通过残差连接、层归一化和前馈网络：

$$
\tilde{\mathbf E}^{(c)}=
\operatorname{LN}\!\left(\mathbf E^{(c-1)}+\operatorname{MHA}(\mathbf E^{(c-1)})\right),
$$

$$
\mathbf E^{(c)}=
\operatorname{LN}\!\left(\tilde{\mathbf E}^{(c)}+\operatorname{FFN}(\tilde{\mathbf E}^{(c)})\right),
$$

$$
\operatorname{FFN}(\mathbf z)=\mathbf W_2\operatorname{ReLU}(\mathbf W_1\mathbf z+\mathbf b_1)+\mathbf b_2.
$$

当前基线使用 1 层 Transformer 编码器，前馈隐层维度为 96，dropout 为 0。编码后采用掩码平均池化获得整个方案的表示：

$$
\mathbf h_l=
\frac{\sum_{i=1}^{M_{\max}}m_i\mathbf e_i^{(C)}}
{\max\left(1,\sum_{i=1}^{M_{\max}}m_i\right)}
\in\mathbb R^{48}.
$$

当前 `full_transformer` 配置直接使用 $\mathbf h_l$。代码同时预留 `hybrid_transformer_fc` 模式，可将全局 token 另经 MLP 映射后与 $\mathbf h_l$ 拼接，但该模式不是当前默认实验设置。

### 3.3 策略网络及其输出

策略网络以共享方案表示 $\mathbf h_l$ 为输入，设置四个并行分类头，分别输出算子编号和三个操作对象编号的 logits：

$$
\begin{bmatrix}
\mathbf z_l^{\mathrm{op}}\\
\mathbf z_l^{(1)}\\
\mathbf z_l^{(2)}\\
\mathbf z_l^{(3)}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf W_{\mathrm{op}}\\
\mathbf W_1\\
\mathbf W_2\\
\mathbf W_3
\end{bmatrix}\mathbf h_l+
\begin{bmatrix}
\mathbf b_{\mathrm{op}}\\
\mathbf b_1\\
\mathbf b_2\\
\mathbf b_3
\end{bmatrix}.
$$

其中，$\mathbf z_l^{\mathrm{op}}\in\mathbb R^8$，三个对象头均为 $\mathbb R^{512}$。四个头分别经 Softmax 形成分类分布，并独立采样得到

$$
a_l=(o_l,u_l^{(1)},u_l^{(2)},u_l^{(3)}),
$$

$$
\pi_\theta(a_l\mid s_l)=
\pi_\theta(o_l\mid s_l)
\prod_{q=1}^{3}\pi_\theta(u_l^{(q)}\mid s_l),
\qquad
\log\pi_\theta(a_l\mid s_l)=
\log\pi_\theta(o_l\mid s_l)+\sum_{q=1}^{3}\log\pi_\theta(u_l^{(q)}\mid s_l).
$$

八类邻域算子包括节点重定位、节点交换、2-opt、处理处置节点替换、服务周期移动、增加提前服务、按危化废弃物类型拆分路线和单节点拆分路线。三个对象编号会根据算子语义映射为访问节点、路线、插入位置、周期、车辆、处理处置节点或危化废弃物类型；当实际对象数小于 512 时，当前实现通过模运算映射到有效对象集合。因此，网络输出的不是“下一访问节点”，而是对完整初始解实施局部改进所需的复合动作。

### 3.4 价值网络及其输出

策略网络和价值网络共享 token 投影、Transformer 编码器及池化层，价值分支在共享表示上使用一个双输出线性头：

$$
\mathbf V_\psi(s_l,\boldsymbol\omega)=
\begin{bmatrix}
V_{\psi,c}(s_l,\boldsymbol\omega)\\
V_{\psi,r}(s_l,\boldsymbol\omega)
\end{bmatrix}
=\mathbf W_V\mathbf h_l+\mathbf b_V\in\mathbb R^2.
$$

两个分量分别估计归一化成本奖励和归一化风险奖励的期望累计回报。保留向量 Critic 而不是直接预测单一加权价值，可避免风险改善信号因量纲或偏好权重较小而被成本分量掩盖，也允许同一网络适配不同的成本—风险偏好。

### 3.5 两类 mask 与当前实现边界

需要区分“输入 padding mask”和“动作可行性 mask”。当前代码完整实现的是前者。令

$$
m_i=
\begin{cases}
1,&i\le M_l,\\
0,&i>M_l,
\end{cases}
$$

则注意力中的加性掩码为

$$
B_{ij}=
\begin{cases}
0,&m_j=1,\\
-\infty,&m_j=0.
\end{cases}
$$

因此，补零 token 既不能作为有效的 Key/Value 参与注意力，也不会进入最终的平均池化。

当前 Actor logits 尚未设置逐算子的不可行动作硬掩码，即没有在 Softmax 前把不适用的算子或对象位置为 $-\infty$。动作约束通过“执行—修复—验证”链处理：首先按对象编号应用邻域算子，然后在车辆容量、处理技术、危化废弃物兼容性、弧可达性、跨期库存传播和末期库存清零等约束下确定性修复；若动作没有产生变化或修复失败，则保持原方案并给予负奖励。因而，本文所称 mask 应限定为 Transformer 输入的 padding mask；不可行动作的处理属于后验可行性过滤和惩罚机制。若后续实现动作硬掩码，可定义算子相关可行集 $\mathcal A(s_l)$，并将

$$
\tilde z_a=
\begin{cases}
z_a,&a\in\mathcal A(s_l),\\
-\infty,&a\notin\mathcal A(s_l)
\end{cases}
$$

后再计算 Softmax，但该式仅为可扩展方向，不应表述为当前已实现功能。

## 4 PPO 训练框架与奖励设计

本研究采用 PPO-Clip 训练 Actor-Critic 网络。其主体与标准 PPO 一致：首先使用更新前的策略与环境交互并采集轨迹，然后估计优势函数，最后在同一批轨迹上多轮优化截断代理目标、价值函数损失和熵正则项。项目特有部分主要体现在成本—风险双目标奖励和向量价值网络，而不改变 PPO-Clip 的基本更新原理。

### 4.1 奖励设计

设 $C_l$ 和 $R_l$ 分别为第 $l$ 步方案的总成本和总风险，$C_0$ 和 $R_0$ 为该条轨迹初始解的对应指标。成本和风险的具体组成已在数学模型部分定义，此处仅说明强化学习使用的即时奖励。给定成本—风险偏好 $\boldsymbol\omega=(\omega_c,\omega_r)^\mathsf T$，对成功执行并通过修复验证的动作，奖励定义为相邻两步加权目标值的相对改善量：

$$
r_l^{\boldsymbol\omega}
=\omega_c\frac{C(\mathcal P_l)-C(\mathcal P_{l+1})}{C_0}
+\omega_r\frac{R(\mathcal P_l)-R(\mathcal P_{l+1})}{R_0}.
$$

目标下降时获得正奖励，目标上升时获得负奖励；用初始解指标作为尺度，可以缓解成本与风险的数量级差异。若动作虽通过验证但未改变两个目标，则施加轻微惩罚；若动作未产生有效变化或修复失败，则施加更大的固定惩罚。通过验证的候选即使使目标变差，也会作为下一训练状态并通过负奖励反馈；只有无效或修复失败的动作保持原状态。训练时为每条轨迹采样一个成本—风险偏好，使同一策略能够学习不同偏好下的改进方向。上述设计属于本问题的奖励塑形，不是对 PPO 更新规则的修改。

> **实现说明（供写作参考）：** 为训练成本—风险双输出 Critic，代码内部仍分别保留上述公式中的成本改善项 $r_l^c$ 和风险改善项 $r_l^r$，并对两者分别计算 GAE；正文奖励函数直接写成标量 $r_l^{\boldsymbol\omega}$ 即可。

### 4.2 优势函数估计

PPO 使用广义优势估计（Generalized Advantage Estimation, GAE）在估计偏差与方差之间取得平衡。由于当前价值网络分别输出成本价值和风险价值，对 $q\in\{c,r\}$，其时序差分残差和优势估计写为

$$
\delta_l^q=r_l^q+\gamma V_q(s_{l+1})-V_q(s_l),
$$

$$
A_l^q=\sum_{d=0}^{L-l-1}(\gamma\lambda)^d\delta_{l+d}^q,
\qquad
\hat G_l^q=A_l^q+V_q(s_l),
$$

其中，$\gamma$ 为折扣因子，$\lambda$ 为 GAE 参数，$\hat G_l^q$ 为价值网络的回报目标。实现中先分别标准化成本优势和风险优势，再依据当前偏好合成为 Actor 使用的标量优势：

$$
A_l^{\boldsymbol\omega}=
\omega_c\operatorname{Norm}(A_l^c)
+\omega_r\operatorname{Norm}(A_l^r),
$$

合成后的优势再次进行标准化，以提高批训练的数值稳定性。这种“分目标估计、按偏好合成”的处理是标准 GAE 在多目标情形下的扩展。

### 4.3 PPO-Clip 更新目标

记采集轨迹时的行为策略为 $\pi_{\theta_{\mathrm{old}}}$，待更新策略为 $\pi_\theta$。当前实现保存采样动作在行为策略下的对数概率，并在参数更新时计算新旧策略的联合概率比：

$$
\rho_l(\theta)=
\frac{\pi_\theta(a_l\mid s_l)}
{\pi_{\theta_{\mathrm{old}}}(a_l\mid s_l)}
=\exp\!\left[
\log\pi_\theta(a_l\mid s_l)-
\log\pi_{\theta_{\mathrm{old}}}(a_l\mid s_l)
\right].
$$

PPO-Clip 通过截断概率比限制单次策略更新幅度，Actor 损失为

$$
\mathcal L_{\mathrm{actor}}(\theta)=
-\mathbb E_l\left[
\min\left(
\rho_l(\theta)A_l^{\boldsymbol\omega},
\operatorname{clip}(\rho_l(\theta),1-\epsilon,1+\epsilon)
\,A_l^{\boldsymbol\omega}
\right)
\right],
$$

其中，$\epsilon$ 为截断系数。Critic 通过均方误差拟合成本和风险的回报目标：

$$
\mathcal L_{\mathrm{value}}(\psi)=
\mathbb E_l\left[
\left\|\hat{\mathbf G}_l-\mathbf V_\psi(s_l,\boldsymbol\omega)\right\|_2^2
\right],
$$

为保持探索能力，在损失函数中加入复合动作各分类分布的熵奖励。Actor-Critic 的联合损失为

$$
\mathcal L=
\mathcal L_{\mathrm{actor}}
+c_V\mathcal L_{\mathrm{value}}
-c_H\mathbb E_l[\mathcal H_l],
$$

其中，$\mathcal H_l$ 为算子头和三个对象头的熵之和，$c_V$ 和 $c_H$ 分别为价值损失系数和熵系数。该形式对应标准 PPO-Clip；当前实现没有采用 KL 散度惩罚项。

### 4.4 训练流程

PPO 训练过程概括如下。

1. 初始化 Actor-Critic 网络，并为各并行环境构造初始可行解和目标偏好。
2. 使用当前行为策略采样复合动作，与环境交互并记录状态、动作、即时奖励、价值估计和旧策略对数概率。
3. 轨迹采集完成后，分别计算成本和风险的 GAE 及回报目标，再按偏好合成 Actor 优势。
4. 固定轨迹数据和旧策略对数概率，多轮计算截断策略损失、价值损失和熵正则项，并使用 Adam 更新网络参数。
5. 重复“采集轨迹—更新参数”过程，直至完成预设训练迭代，并保存网络参数与复现实验所需配置。

当前 PPO 训练参数如下：

| 项目 | 数值 |
|---|---:|
| 并行 episode / 每条轨迹步数 | 24 / 24 |
| PPO 更新轮次 | 24 |
| $\gamma$ / $\lambda$ / $\epsilon$ | 0.95 / 0.9 / 0.2 |
| 学习率 | $3\times10^{-4}$ |
| 价值损失系数 / 熵系数 | 0.5 / 0.02 |
| 无效或 repair 失败惩罚 / 有效但目标无变化惩罚 | 0.2 / 0.01 |

### 4.5 实现核对说明（供写作参考）

> 本小节用于区分通用 PPO 与当前代码实践，不建议原样放入论文正文。正文可概括为：“本文遵循标准 PPO-Clip 更新框架，并针对成本—风险双目标和复合邻域动作进行了适配。”

总体而言，当前训练代码实现的是标准 PPO-Clip 主流程：使用行为策略采集 on-policy 轨迹，固定旧动作对数概率，采用 GAE、截断代理目标、价值均方误差、熵正则、多轮更新、Adam 优化和梯度裁剪。没有单独复制一套旧策略网络并不构成算法差异，因为更新期间保存并固定旧策略对数概率即可计算 PPO 概率比。

当前实践仍有以下项目扩展或实现差异，需要在解释实验结果时加以区分。

1. **多目标扩展。** 标准 PPO 通常使用标量奖励和标量 Critic；当前实现使用成本—风险二维奖励和二维 Critic，先分别估计并标准化两个优势分量，再按偏好合成为 Actor 优势，且合成后的优势会再次标准化。这是本项目最主要的算法扩展。
2. **复合动作分解。** 动作概率由算子头和三个对象头的分类概率相乘得到。使用四个分类头对数概率之和计算联合概率比，因此该动作参数化与 PPO-Clip 兼容，但不同于标准示例中的单一离散动作。
3. **轨迹末端处理。** 当前代码将每条固定长度轨迹末端的下一状态价值直接置零，且没有区分真实终止与时间截断。如果 24 个改进步仅是采样时域截断而不是真实终止，则标准 GAE 通常应使用末状态价值进行 bootstrap；当前处理会使靠近轨迹末端的回报估计产生额外偏差。
4. **更新粒度。** 当前 micro-batch 仅用于显存受限时的梯度累积，每个更新轮次遍历全部样本后才执行一次优化器更新；常见 PPO 实现则在每个随机 mini-batch 后更新一次参数。因此，当前实现更接近“每轮一次全批量梯度更新”，但截断代理目标本身仍是 PPO-Clip。
5. **可选稳定机制。** 当前实现没有价值函数裁剪、目标 KL 早停或显式 `done` 掩码。前两项是常见但非 PPO-Clip 必需机制；`done/truncated` 区分则与第 3 点的 bootstrap 问题直接相关。

> **写作参考说明（定稿时可删除）：** 王苓等文献按照“奖励—新旧策略概率比—Actor 截断损失—Critic 损失—训练步骤及超参数”的顺序介绍 PPO，本文沿用这一清晰结构。该文同时讨论了 KL 约束和截断目标，而当前代码只实现 PPO-Clip，因此正文不应声称使用了 KL 惩罚或 KL 约束。

## 5 训练后求解过程

训练完成后，算法仍从给定初始可行解出发。在每个改进步，Actor 首先给出一个四头联合贪心动作，再从策略分布中采样若干动作；当前配置每步共考察 32 个候选。所有候选均需通过算子执行和 repair，随后按

$$
J_{\boldsymbol\omega}(\mathcal P)=
\omega_c\frac{C(\mathcal P)}{C_0}
+\omega_r\frac{R(\mathcal P)}{R_0}
$$

选出本步最优可行候选作为后续搜索状态。同时，算法单独维护历史最优 incumbent：只有当候选目标严格优于 incumbent 时才更新最优解，最终返回该 incumbent，而不是最后一次搜索状态。这一机制允许策略暂时经过非改进状态探索新的邻域，又保证最终交付方案不劣于搜索过程中已发现的最好方案。

## 6 方法特点与表述边界

本文 PPO-Transformer 的核心可概括为：以初始可行方案为输入，将多周期路线、产废节点与处理处置节点库存、处理处置节点状态、弧风险和成本—风险偏好加工为异构 token；利用 Transformer 建模不同实体及不同周期之间的全局关联；由共享策略网络—价值网络同时输出复合动作和成本—风险双价值；最后通过偏好加权的标量改善奖励、偏好条件 GAE 与 PPO 截断目标学习邻域改进策略。

与参考文献中的方法相比，两者都采用“分类特征分别投影—多头注意力编码—策略输出—约束处理—强化学习训练”的叙述链条，但决策机制存在本质区别：参考文献从节点特征出发，使用 Encoder-Decoder 和节点掩码逐步构造多车辆路线，并以 REINFORCE 训练；本文则从完整初始解出发，使用 Transformer Encoder 汇聚方案状态，输出邻域算子及其对象，并以 PPO 训练。因而，不应将参考文献中的车辆轮换解码器、下一节点硬掩码或滚动基准网络直接归入本文实现。

---

**实现核对依据（定稿嵌入正文时可删除）：** token 构造、Transformer Actor-Critic 与 PPO 训练见 `src/ppo_improver.py`；八类算子和 repair 见 `src/operators.py`；初始解见 `src/heuristics.py`；成本、风险及统一评价见 `src/solution_utils.py`；超参数见 `configs/network_config.json` 与 `configs/algorithm_config.json`。token 部分的写作结构参考《众包模式下基于并行生成策略的强化学习路径规划模型》PDF 第 5—9 页及附录训练算法；PPO 部分的表达层次参考《面向连续泊位和岸桥动态调度问题的强化学习方法》PDF 第 4—5 页。参考文献仅用于组织逻辑和详略控制，公式与机制均按当前代码及标准 PPO-Clip 写法核对后重写。
