# PickXtimes 服务器后台训练与评测

这个脚本在Linux服务器运行。默认等待已有Stage1进程`1776511`，每600秒检查一次；同一进程仍在运行时不会启动新的训练。进程结束后核验`runs/ckpts/pickxtimes_stage1_6gpu/20000`是完整、已完成20,000步的Stage1快照，再进入后续队列。

默认队列为：

1. 等待GPU1–6空闲，初始化并训练Stage2，4轮交替，每轮500次记忆/门控更新和500次策略更新，共4,000次。
2. 评测A：自己的Stage1 `/20000`，关闭记忆。
3. 评测B：Stage2最终快照，开启完整记忆。
4. 评测C：使用同一个B快照，推理时关闭所有记忆写入。
5. 校验三组评测协议、初始观察和模型来源，输出比较表。

Stage2使用A快照中的配置，沿用当前六卡实验的完整历史、batch32和normalization，初始化来源仍明确指向自己的A。每5,000步和结束时保存，因此默认4,000步的B只在结束时保存。4轮是本次实验预算，论文未公布总轮数。

评测依次执行，每模型50个官方test episodes × seeds7/17/27，共150次rollout。默认policy和simulator都使用物理GPU6，沿用此前已运行的单卡仿真路径；训练结束后会释放六卡训练进程，再启动单卡policy。需要修改设备时可用`--devices`、`--policy-gpu`、`--sim-gpu`，具体选项见`--help`。

## 启动

先将本地代码自行commit/push，服务器`git pull`后执行：

```bash
cd /data0/workspace/gaoyuxuan2/robomme_policy_learning
git pull
mkdir -p runs/logs

nohup bash scripts/t2mem/run_pipeline.sh \
  --watch-pid 1776511 --poll-interval 600 \
  > runs/logs/pickxtimes_pipeline.log 2>&1 &

printf '启动PID=%s\n' "$!"
tail -n 60 -f runs/logs/pickxtimes_pipeline.log
```

launcher会读取已有`$HOME/.config/t2mem/env.sh`，在仓库内使用现有policy环境运行`uv run --no-sync`。不需要sudo、cron或保持SSH连接。主程序只使用标准库做进程管理；JAX训练、policy和仿真分别由子进程在服务器执行。

启动时监控PID已经结束也可以继续：脚本先检查A最终快照。`metadata.json`必须标记快照完整和训练预算完成，且params、train_state、normalization存在；只看到PID消失或一个未完成保存目录不会自动启动B。PID的UID和Linux启动时间用于区别同一数字被系统复用的进程。

同一队列使用文件锁防止重复启动。GPU排队检查NVIDIA实际compute进程，空闲后启动工作；它不是集群调度器的硬件预约。端口8090须空闲，policy就绪检查核验websocket返回的快照、A/B/C模式、H64/E32、stride16及normalization，而不只检查端口是否监听。

## 日志与结果

默认流水线目录是`runs/pipeline/pickxtimes_6gpu`：

- `state.json`：当前阶段、驱动进程PID、监控进程身份、完成/失败状态。
- `driver.pid`：实际Python驱动PID；shell打印的`nohup`PID可能属于uv父进程。
- `logs/`：Stage2、各policy server、各评测和汇总的独立日志。
- `evaluation/A`、`evaluation/B`、`evaluation/C`：逐episode记录和成功率。
- `comparison/`：CSV、Markdown比较表及来源/协议校验信息。

查看状态和结果：

```bash
cat runs/pipeline/pickxtimes_6gpu/state.json
tail -n 80 -f runs/logs/pickxtimes_pipeline.log
cat runs/pipeline/pickxtimes_6gpu/comparison/comparison.md
```

关闭`tail`只停止查看日志，后台队列继续运行。明确需要停止整个自动队列时，可向`driver.pid`中的进程发送TERM；它只清理本脚本创建的训练、policy和评测子进程，不会终止仍在监控的原Stage1进程。

## 失败与重启

进程异常退出、快照不完整、来源不匹配或训练返回非零，写入失败状态后停止队列。已有Stage2完整快照可用于恢复尚未完成的训练；没有可恢复快照的非空训练目录会报错并保留原文件，避免把已有训练结果覆盖掉。

评测遇到基础设施错误会在同一目录最多尝试3次，已完成的有效rollouts不重跑；任务自身的`timeout`是有效失败，可以正常进入成功率统计。全部150次有效rollouts、无基础设施错误、难度数量符合26/12/12才算正式评测完成。policy异常退出也会停止相应评测进程；默认连续3,600秒没有新的episode记录时终止本轮并重试，可用`--eval-idle-timeout`调整。已完整且来源一致的评测可以跳过，来源不同则报错。

用相同启动命令重启驱动时会检查已有Stage2快照和评测记录，继续可恢复的阶段。GPU训练和正式评测仍需在服务器验证；本地仅进行语法检查和使用临时文件、模拟进程的标准库回归。

单独验证控制逻辑，无需模型、数据下载或GPU：

```bash
uv run --no-sync python tests/test_t2mem_pipeline.py
```
