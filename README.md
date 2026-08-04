# Agent1

一个用于多人协作开发的 Python Agent 项目。

## 项目结构

```text
agent-project/
  app/
    __init__.py
    main.py
  .gitignore
  README.md
  requirements.txt
```

## 环境要求

- Python 3.10+
- Git
- PyCharm

## 安装步骤

克隆项目：

```bash
git clone https://github.com/yuuu136/agent1.git
cd agent1
```

创建虚拟环境：

```bash
python -m venv .venv
```

Windows 激活虚拟环境：

```bash
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
python -m app.main
```

如果看到：

```text
Agent project started
```

说明项目运行成功。

## Git 协作流程

主分支说明：

```text
main        稳定版本
dev         日常开发分支
feature/*   功能开发分支
```

开发新功能时：

```bash
git checkout dev
git pull
git checkout -b feature/your-feature-name
```

提交代码：

```bash
git add .
git commit -m "Describe your change"
git push -u origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request，合并到 `dev` 分支。

## 注意事项

不要提交以下内容：

- `.venv/`
- `.env`
- API Key
- 临时文件
- 本地 IDE 缓存

如果需要环境变量，请创建 `.env` 文件，并不要上传到 GitHub。