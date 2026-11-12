# CN Daily-Recurrent 改造与验证报告

验证日期：2026-08-31。未启动正式训练，无正式 checkpoint 或性能结论。

## 目录与边界

按本次聊天要求新建独立目录：
`E:\pythonpro\NEW_DRFN\N2M_Filtered_BidirPCDA_DailyRecurrent_CN`。

原 `N2M_Filtered_BidirPCDA_Fixed_CN` 和美股参考工程均未修改。
两处共 32 个原始源文件/文档的 SHA256 前后完全相同。
新工程不包含旧 checkpoint、旧日志或旧模型副本。

## 最终文件树

```text
N2M_Filtered_BidirPCDA_DailyRecurrent_CN/
  Layers_n2m_allstocks_dailyrecurrent.py
  Model_n2m_allstocks_dailyrecurrent.py
  main_n2m_allstocks_dailyrecurrent.py
  run_n2m_allstocks_dailyrecurrent_cn.py
  backtesting.py
  data_preprocessing.py
  data_splits.py
  utils.py
  project_paths.py
  requirements.txt
  .gitignore
  cn_price.csv
  cn_price_normalized.csv
  cn_target.csv
  cn_news.csv
  Tinybert-4layers/                  原始中文资源，7 个文件
  cn_asset_contract.json
  core_math_contract.json
  verify_cn_data.py
  verify_package.py
  verify_fixed_implementation.py
  verify_dailyrecurrent_cn.py
  smoke_forward_fixed.py
  smoke_dailyrecurrent_cn.py
  README_DailyRecurrent_CN.md
  VALIDATION_DailyRecurrent_CN.md
  validation/smoke_cpu/              仅短流程证据，不是正式实验结果
```

正式运行时自行创建 `SavedModels/<exp-name>`、`result/<exp-name>`、
`logs/<exp-name>`。验证脚本名称中的 fixed 不代表另一套模型；
正式工程只有一套 Model/Layers/main/runner。

## 代码改动

- 原四个 *_pilot 核心文件在新工程中改为 *_dailyrecurrent；
  imports、runner、回测、测试和文档全部同步。
- Model 增加日度隐藏状态及日期检查；GRU_r 首次 5 天，此后仅新增终点日。
  GRU_s 保持每个窗口从零开始的滚动 5 天预测分支。
- main 的 PCDA 改为上一交易日逐头关系监督当前终点日；
  使用 16 日 TBPTT 分块均值反传，step 后只 detach，不清隐藏值。
- 日度统计字段、实际阶段 manifest、输出路径隔离同步更新。
- backtesting 保留三种显式关系记忆干涉，并更新独立权重加载后的日序推理。
- data_splits 保留 A 股七阶段算法，只删除无调用的单次划分接口。
- data_preprocessing 只更新错误的市场说明和运行示例；静态图函数数学未变。
- utils 保留原 A 股的 seed、初始化、strict target 和指标逻辑。
- 新增中文资产解析、数据校验合同、数值/时序测试及本报告。

## 数据校验

| 文件 | 修改前与修改后 SHA256 | 对比 |
|---|---|---|
| cn_price.csv | `43402975aa3ad125d93e47bd64a441526ec7bc0711228dd1f0f15e046bc350d0` | 相同 |
| cn_price_normalized.csv | `26d026437178023f2864f8263b1d62dbfe7bf0f1f0c168fa1a50df227156643a` | 相同 |
| cn_target.csv | `c38476e4464753c2abe733444abb2c6cb9a065762eea4084eca07a19bb035f90` | 相同 |
| cn_news.csv | `9d40e11546c1ad0368bec29664338e49134f88806009d4d06365870f38d1c1d4` | 相同 |

四表均为完整的 160 股票 × 1942 日期 = 310720 行，无重复股票日期键。
目标收益率 310560 个有限值，仅最后一天 160 个 NaN。
重新计算的 t→t+1 target 最大绝对误差为 2.1e-16。
五项行情的逐日横截面 sample z-score（ddof=1）误差均不超过 1.78e-15。
没有重写 CSV，也没有修改股票池、目标或归一化定义。

## 中文 TinyBERT

`Tinybert-4layers` 的全部 7 个文件 SHA256 均与原目录相同；
详细值保存在 `cn_asset_contract.json`。

配置为 4 层、hidden 312、模型及 tokenizer 词表 21128。
预训练参数保留、完全冻结、eval 模式。测试确认 DRFN 初始化和
model.train() 均不重置或启用 TinyBERT 训练；错误的英文词表会被拒绝。
保留实际新闻长度打包、无新闻日零输出及窗口内后续新闻日的 hidden 延续。

## 时序证据

实际 P1 训练起点：

| 当前终点日 | 上一次关系日期 | GRU_r 输入天数 | 本次处理日期 | PCDA 来源 |
|---|---|---:|---|---|
| 2018-01-08 | 无 | 5 | 01-02、01-03、01-04、01-05、01-08 | 无 |
| 2018-01-09 | 2018-01-08 | 1 | 01-09 | D_01-08 → 01-09 |
| 2018-01-10 | 2018-01-09 | 1 | 01-10 | D_01-09 → 01-10 |

上一日 PCDA 输入维度为 [8,160,160]；
当前日 news/market 对齐表示均为 [1,160,64]。
首次无历史动态关系，因此没有 PCDA 对；随后使用 detach 后的逐头关系。

TBPTT 第一块 16 日在 2018-01-29 完成 backward、clip、step、zero_grad、
hidden detach；下一日使用保留的 hidden 数值且已切断之前的图。
余下第 17 日于 2018-01-30 单独做实际长度为 1 的均值更新。
每次短训练调用均断言两次 detach；不把 remainder 再除以 16。

实际检查 schedule epoch 0 和 5：
epoch 0 的系数为 0，但关系正常递归；
epoch 5 的系数为 2e-5，训练和验证均实际调用当前日 PCDA。
这不是缩短正式 warmup，而是短测试直接覆盖两个调度分支。

## 阶段日期

七阶段严格测试预测日数：
`[163, 160, 158, 163, 161, 157, 164]`。

P1：原始测试行情截止 2021-12-31，合法预测截止 2021-12-30。
P7：训练 2022-01～2024-12；验证 2025-01～04；测试 2025-05～12。
P7 最后合法预测 2025-12-30，对应下一全局交易日 2025-12-31，
两者都在 P7 测试期内，共 164 个合法预测日。
全部阶段的 train/val/test 都检查了 next-global-day 同 split 条件。

## 不变部分审计

58 个受保护定义通过 AST 校验，涵盖 CN 编码器、清洗、基础对齐、
PCDA 数学、静态关系、逐头 Dynamic/Relative Static/Fusion、
传播、残差、严格日期和回测指标；对已删除的无效参数只做明确规范化。
新增日度状态及 TBPTT 与参考日度实现逐定义核对。

另以相同种子实际实例化原 CN 与新模型：
保留的 194 个 state_dict 条目逐项完全一致，构造时 RNG 状态一致，
只去掉 18 个没有参与计算的 lamda/beta 标量及逐头别名。
首个 bootstrap 窗口六项输出完全相同；后续时间递归是本次有意改变的部分。

activation checkpoint 开/关测试确认输出、全部参数梯度、RNG、
隐藏状态和逐头显式记忆一致，重算不额外推进日期。
保留 proj_residual、log_margin 等深层参数。

## 干涉与输出

三个模式均完成真实五日回测推理：
Full、No Explicit Relation Memory、Shuffled Explicit Relation Memory。
单元测试确认 No Explicit 和 Shuffled 都不清隐藏状态；
打乱对所有 head 使用同一个股票顺序同时作用于行与列。
每次独立推理首尾 reset，GRU_r 步数均为 5、1、1、1、1。

正式 Sharpe、年化、回撤、预测及排序指标公式未改。
prediction.csv 的九字段和 seed/phase 替换机制已检查；
metrics.csv 能由 canonical prediction 重算预测/排序指标。
Precision@5 在 pred>0 交易过滤之前计算。
保留三套 memory 指标及 Full 相对干涉版本的 drop。

## 清理

删除的正式接口/路径：第四项 dynamic soft loss、dynamic_thresh、
D_dynamic_prev、lamda/beta/infer、旧动态 hard-alignment helper、
耦合分析函数、无调用的单次划分 API。
日统计统一 per_day，不保留 per_window 别名。
没有引入 sensitivity 或 last_fused_relation。
正式源码/README 无错误的英文 TinyBERT、116 股票或九阶段设置。
未复制原目录历史 result/logs/SavedModels；本次 smoke 证据单独归档。

## 验证结果与限制

| 检查 | 结果 |
|---|---|
| compileall 与 15 个 Python 模块导入 | PASS |
| verify_cn_data | PASS |
| verify_package | PASS |
| verify_fixed_implementation | PASS |
| verify_dailyrecurrent_cn | PASS |
| CPU 完整规模短流程 | PASS |
| CUDA 单窗口前向/反向 | PASS，RTX 3050 8GB，cuda:0 |
| 原始 CN/参考 US 源文件前后校验 | PASS |

CPU 短流程：160 股票、8 头、原隐藏维度、TBPTT=16，
两个 17 日训练调用、各 5 日验证、3×5 日干涉推理，
40 次当前日 PCDA 调用，临时 checkpoint 保存/重载通过且已删除。
该短样本的收益数值不用于评价模型效果。

CUDA 只验证了完整模型单窗口，不代表完整 16 日 TBPTT 一定能在这张卡装下；
本次没有进行 GPU 完整 TBPTT 或正式训练，也没有修改模型规模来规避显存。
运行环境：Python 3.10.20、PyTorch 2.10.0+cu128、Transformers 4.30.2、
CUDA runtime 12.8、cuDNN 91002。

## 正式 P1 命令（尚未执行）

在新工程目录并激活目标环境后：

```powershell
python run_n2m_allstocks_dailyrecurrent_cn.py --phases "1" --device cuda:0 --seed 11380 --lambda-align 1.0 --max-epoch 300 --wait-epoch 30 --exp-name "cn_dailyrecurrent_phase1_seed11380"
```

不要用原窗口级模型的 checkpoint 代替新架构正式训练。

