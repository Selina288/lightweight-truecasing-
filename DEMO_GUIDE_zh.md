# 前端演示启动说明

这是本地网页界面，调用真实的Python模型，不是静态模拟，也没有部署到公网。

## 第一次启动

在 lightweight-truecasing-frontend 文件夹打开终端（也可以输入 cd 加空格，再把文件夹拖进终端后回车），逐条运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/demo_server.py
```

看到 Truecasing demo: http://127.0.0.1:8765 后，在浏览器地址栏输入：

http://127.0.0.1:8765

不要双击 web/index.html；需要Python服务器才能运行预测。

## 页面怎么使用

1. 输入英文，或点击一个示例。
2. 选择模型：C 为上下文加字符版；B 为上下文版；另外有三个baseline。
3. 点击 Restore capitalization。
4. 输出区显示预测结果，浅绿色高亮大小写变化的词。可以复制结果。
5. 下方分数是先前保存的 dev 实验结果，不是当前输入的准确率。

输入会先统一转成小写。简单分词与正式实验的UD分词不同，缩写和缩合形式可能造成差异。标点保持原样，不会自动补标点。

## 后续再次启动

在相同项目目录运行：

```bash
source .venv/bin/activate
python src/demo_server.py
```

终端窗口需要保持打开。停止时按 Ctrl+C。如果8765端口已被占用，可执行 `python src/demo_server.py --port 8766`，浏览器相应打开 http://127.0.0.1:8766。

## 模型文件与GitHub

完整包包含训练好的B/C模型和baseline词典所需的train数据；GitHub更新包不含这些本地文件。别人从GitHub下载后需要获取原始数据，依次运行 `python src/preprocess.py` 和 `python src/ablation.py` 来生成数据和模型。缺少模型时，页面会将相应选项标为不可用，两个无需数据的baseline仍可使用。

## 已验证与后续工作

22个自动测试通过；五种模型的HTTP调用通过；网页中C模型和全小写baseline切换、真实输出和高亮已检查。没有读取test，也没有改变已有实验结果。组内仍需确认19个助手复核例子，冻结最终方法后再运行test，并完成论文和展示。
