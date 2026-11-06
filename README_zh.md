[English](README.md) | **简体中文**

# 可审计报警处置

## 可执行证据角色隔离与非对称风险校准

论文：**Auditable Alarm Disposition through Executable Evidence Role Separation and Asymmetric Risk Calibration**
作者：Wu Zhang、Tianwu Lei、Luo Xiao、Yong Yang。研究修订日期：2026 年 11 月 6 日。

工业报警处置需要结合传感器测量、历史案例和语言模型解释。引用有效的当前证据，并不意味着模型的判断不会受误导性历史影响。本项目将数值处置与历史解释分离，并分别校准两类方向性错误。

## 核心方法

| 对象 | 作用 |
| --- | --- |
| 当前证据 `E` | 事件传感器特征及训练阶段拟合的变换 |
| 历史案例 `H` | 向语言读者提供类比背景 |
| 语言解释 `L = G(E,H)` | 保存带证据引用的读者解释 |
| 数值处置 `R = rho(E; theta)` | 冻结策略输出 FAULT、BENIGN 或 ABSTAIN |

数值路径不接收历史叙述或生成文本。在当前证据与策略固定时，修改历史案例不会改变数值处置。良性与故障阈值分别控制两类错误，在满足校准条件的候选中选择自动覆盖率最高的策略。证据、策略和响应绑定支持审计与数值回放。可选的读者审计放行机制与独立数值核心分开评价。

## 主要实验结果

| 实验 | 主要结果 |
| --- | --- |
| 误导性历史干预 | 普通读者故障转良性错误 5→20；角色隔离读者 3→12；独立数值路径在两种条件下均为 2/20 |
| TEP 分组选择性校准 | 15/15 次评价满足双向错误标准，平均自动覆盖率 87.7% |
| HAI 时间顺序适配 | 检出 17/17 个攻击事件，自动覆盖率 98.5%；窗口级故障转良性错误 2 个、正常转故障错误 4 个 |

TEP 使用三个随机种子的五折分组评价。HAI 使用更早的数据拟合、test3 校准、test4/test5 的 1100 个窗口评价。上述结果来自回顾性基准实验。解释质量采用 MiniMax-M3 自动评价，本修订版不主张人工评价结果。

## 代码与使用

- [研究实现与阶段说明](research/README.md)：重建实验源码、校准、时间顺序适配、历史干预、审计和补充分析。
- [早期演示包](src/README.md)：合成示例与可插拔接口；其中的代理检测器和 ACSC 示例与修订版基准实现分开保留。

```sh
python -m venv .venv
# 按当前终端方式激活环境。
python -m pip install -r research/requirements.txt
python research/prepare_workspace.py
```

单独获取公开 TEP 与 HAI 21.03 数据，按照研究目录的阶段说明执行。脚本依赖前序阶段生成的产物，仅安装环境不会自动完成全部实验。

语言读者为 `MiniMax-M3.1-Flash`，自动评价器为 `MiniMax-M3`。API 脚本通过环境变量 `MINIMAX_API_KEY`、`MINIMAX_API_KEY_2` 和 `MINIMAX_API_KEY_3` 获取凭据，每路六并发，新请求不设置客户端 token 上限。输入产物齐备后，离线数值分析不需要调用 API。

## 仓库资料

| 目录 | 内容 |
| --- | --- |
| `research/` | 研究脚本、依赖版本及运行目录初始化 |
| `src/` | 早期演示包、示例和测试 |
| `manuscript/` | 八月版论文 PDF 归档，早于本页介绍的修订版 |
| `assets/` | 归档论文配图 |

旧 PDF 与配图不代表当前实验结果。本次代码发布未包含最新版论文文件。实验日志、API 原始记录、内部日报、数据、模型权重和结果压缩包不纳入发布。近期论文整理保留九张正文图，并调整摘要、方法与实验分工及分页。

## 引用

```bibtex
@unpublished{zhang2026auditable_alarm,
  title = {Auditable Alarm Disposition through Executable Evidence Role Separation and Asymmetric Risk Calibration},
  author = {Zhang, Wu and Lei, Tianwu and Xiao, Luo and Yang, Yong},
  year = {2026},
  note = {Revised research manuscript}
}
```
