# 探索前回退点

保存于2026-09-09，基于Git HEAD `3d2c6391d9d2b932591e5a8080176d6877ea13cc`和当时未提交的本地文件，不只是Git提交快照。

- `code.zip`：当前src、tests、configs、docs、遗传算法目录，以及根目录Python/Markdown/配置等源文件；不含数据结果库或Git数据库。
- 压缩包SHA256：`d3a051b1ac3a03978097bc89e2fc46d6b6ec567f8473a7c7a151556342f9f8f9`。
- `models/v4_small.pt`：`b83f71560f6afb15fec41f1b05d17866c3490c79b985dfb51e4949d6ba46f82e`。
- `models/v4_large.pt`：`61db3f4dcc6aa768f22dd36515449416facb603b934c9eaf9a5e0278838fab3a`。
- `models/v5_large_failed_comparison.pt`：`574d50293757caa8d704466fa0c38f953feb295f02485283e3969f409a86914a`。

回退时先把code.zip解压到新目录并核对差异，再按需恢复相应源码；不要在工作区根目录执行递归删除或盲目覆盖。探索后新增模块不属于原快照，需要明确列出并另行归档。模型副本应恢复到登记中相应路径，不把v5失败对照替代v4默认模型。

此备份只提供本地回退点，没有创建或推送Git提交。原数据和全部历史结果继续保留在原目录，通过既有哈希清单核验。
