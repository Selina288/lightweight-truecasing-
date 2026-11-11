# Office hour 后的项目更新

## 本次真正完成了什么

1. 阅读四篇原始论文，核对它们的数据、评价单位、指标表述、baseline 和类别政策，写入 EVALUATION_COMPARISON.md。
2. 增加独立的字符家族消融：B 加前后缀、B 加字符 n-gram，与原有 B/C 比较。新模型仅使用 train 拟合、dev 评价。B/C 使用原冻结模型并复现原 dev 结果。
3. 在正文加入明确标注的后续 dev 小节；原 test 表及原有十张表的内容未更改。
4. 新增三个程序检查，现有 34 项测试全部通过。两次新拟合收敛；四个系统的整体 dev 指标通过独立 sklearn 核验；冻结清单中的所有原始文件哈希仍匹配。

## 新结果都是 dev，不是 test

| 系统 | Dev exact accuracy | Dev macro-F1 | 未见词 exact accuracy |
| --- | ---: | ---: | ---: |
| B：上下文 | 91.43% | 0.5245 | 62.19% |
| B＋前后缀 | 92.26% | 0.5682 | 69.87% |
| B＋字符 n-gram | 92.33% | 0.5474 | 69.99% |
| C：两组字符特征 | 92.26% | 0.5688 | 69.93% |

前后缀单独使用时几乎追平 C 的 macro-F1。n-gram 单独使用时整体准确率稍高，但 macro-F1 更低，说明不能只看多数小写词主导的准确率。前后缀版本比 C 少正确恢复 1 个 dev token；n-gram 版本比 C 多 14 个，但少数类别指标较弱。这些微小差异没有经过显著性检验，不可写成“某组特征毫无作用”或“新模型显著更好”。

这里所说“单独”仍包含 B 的全部词汇、上下文、位置和长度特征，不是只使用几个字符、丢掉 B。两组字符串特征有信息重叠，且不同特征集会影响尺度和正则化；不把它解释为词法机制的因果证明。

## 下一次见老师可以说

> We completed a dev-only character-feature ablation with the same classifier settings and decoder. Prefixes and suffixes alone nearly match the full model on macro-F1. Character n-grams alone give slightly higher exact accuracy but lower macro-F1. We also audited four prior papers and found differences in datasets, baseline conventions and category definitions, so we are not claiming a direct comparison with their published scores. Our original test results and model remain unchanged.

## 尚未完成，不能混称已完成

- 没有在相同训练数据量下重新比较不同 genre。
- 没有下载并运行外部 truecaser；完成的是论文评价协议核对，不是外部系统 benchmark。
- 尚未确认部分前人 F1 的完整 scorer 定义，不能因为名称相同就视为四类 macro-F1。
- 没有新增 test 评估、多随机种子或显著性检验。
- 原 test 在本次设计前已看过，所以这次明确标为探索性 dev 后续实验。

## 文件和更新方式

- 新报告：results/character_ablation/REPORT.md。
- 评价对照：docs/EVALUATION_COMPARISON.md。
- 新脚本：src/character_ablation.py；新模型及 token 预测仍在忽略目录，不上传。
- GitHub 连接只有读取权限，本次更新先准备为可上传文件，不能说线上已更新。
- Overleaf 的 main.tex 已在原文件中修改并通过编译；导入后仍需检查最终页数与表格位置，不能沿用上一版分页结论。

实际成员分工仍未确认，保持在 FINAL_CHECKLIST_zh.md 中待补；不会按原 proposal 猜测已完成的个人贡献。
