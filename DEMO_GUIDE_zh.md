# 本地网页演示

网页通过 Python 服务调用真实模型；没有部署到公网。页面分数是保存的 dev 结果，不是输入句子的准确率，也不是最终 test 分数。

## 已有完整项目和模型

在项目根目录打开终端，逐条运行：

```bash
source .venv/bin/activate
python src/demo_server.py
```

如果这个文件夹没有自己的 .venv，先按下面的环境安装步骤建立。看到启动地址后，在浏览器打开 http://127.0.0.1:8765。终端保持打开，停止时按 Ctrl+C。端口被占用时运行 `python src/demo_server.py --port 8766`，打开相应端口。不要双击 index.html。

## 从 GitHub 首次下载

GitHub 不包含原始数据、处理后的数据或训练权重。没有这些文件时，网页仅能使用两个不依赖数据的 baseline；B/C 等选项会禁用。

原实验使用 Python 3.14.0。先确认 `python3 --version`，在项目根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/verify_team_review.py
```

然后按照 [复现说明](docs/REPRODUCIBILITY.md) 下载官方数据，将三个 CoNLL-U 文件直接放到 `data/raw/UD_English-EWT/`。先校验，匹配后再生成数据和模型：

```bash
python src/check_data.py
python src/preprocess.py
python src/check_data.py --processed
python src/ablation.py --output-dir results/reproduced/ablation
python src/demo_server.py
```

这些训练命令只应在单独的新副本执行，不要覆盖原始冻结项目的模型。若哈希不匹配，应停止并核对数据，不能修改记录的哈希来绕过检查。新训练权重也不保证与原始 joblib 文件字节相同。

## 页面操作

1. 输入英文或选择例句。
2. 选择可用模型；C 是上下文加字符特征，B 是上下文模型。
3. 点击 Restore capitalization。
4. 查看浅绿色标记的大小写变化，或复制结果。

输入先转小写，标点和空格保留。页面分词与正式实验的 UD 分词不同，缩合形式和句界可能有差异。它只预测大小写，不补标点；输入由本地服务处理，不保存。

## 当前实验状态

最终 test 已完成，原实验记录为 27 项测试通过。8 个选定 dev 例子已按用户报告记录为团队确认，其余 11 个未确认。见 [确认状态](docs/TEAM_REVIEW_STATUS.md)。此说明取代早期关于前端、最终测试和全部例子待确认的说法；不重新训练原模型或改动已有分数。
