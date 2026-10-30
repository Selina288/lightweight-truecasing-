# 19个例子的二次复核

已经逐条核对官方dev中的原始形式和上下文，并重新加载对应模型验证B/C输出。19个例子的gold、输出、seen/unseen、句首位置和变化类型全部一致。没有修改gold，没有重训，也没有读取test。

本轮调整了5处解释：

- piano：只能确认C误大写；“标题风格”是可能解释，不是已证明的模型机制。
- eh：C输出EH像缩写式大小写，但不能断言模型真的识别成了缩写。
- law：即时上下文不能显式表示完整的daughter-in-law结构，是潜在限制，不是已证实的错误原因。
- NiMo：C先预测成initial_uppercase，所以不能把Nimo的失败只归因于mixed-case重建器。
- AMAZING：上下文看起来有强调式全大写，但从小写输入恢复原始写作风格不一定有唯一答案。

这些仍是助手上下文复核，不是组员已完成的人工标注，因此team_confirmed继续保留No。示例是定性展示，不代表各种错误在人群或整个数据中的频率。

组员复核时逐行检查：原文和上下文是否相符、解释是否合理、是否同意类别。只有实际确认后才更新team_confirmed，并记录复核人。不能把“与数据一致”误写成“所有语言学解释均已被人类验证”。

原proposal中的in-genre-only训练条件现已补齐并在dev验证，结果见results/ingenre/REPORT.md。当前最终评估准备说明在docs/FINAL_EVALUATION_PLAN.md，尚未运行test。
