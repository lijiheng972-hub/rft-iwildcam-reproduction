## Block 1 :GUI agent 定义对话
### Q1: 什么是 GUI agent? 
- 自主代理人 → 通过 **perceive / plan / execute** UI 操作,完成自然语言指定的任务
- 关键区别于自动化脚本:GUI agent 会**自主规划**,脚本只能跑预设流程 
- 载体:电脑桌面 / 手机 / 浏览器(三种 GUI 环境)



### Q2: GUI agent 的载体分类

GUI agent 按运行环境分 4 大类:

1. **Web agent** — 浏览器,看 DOM,操作 click/type/scroll(WebArena, Mind2Web)
2. **Mobile agent** — 手机 App,看 accessibility tree,操作 tap/swipe(AndroidWorld)
3. **Desktop / Computer-use agent** — 电脑桌面,跨 App,主要靠截图(**OSWorld** — EvoCUA 用的)
4. **Specialized** — 游戏 / 设计软件等专用环境(**NitroGen** 是 game agent)

难度梯度:Web < Mobile < Desktop < (Specialized 因任务而异)

我精读的 EvoCUA 属于 **Desktop** 类,这就是为什么 56.7% 算 SOTA(desktop 难)。

### Q3: Base model 训练 vs Agent harness — 关键区分

**直觉混淆点**:看到"verification"或"sandbox"就以为是 harness。
**真实区分**:这些东西**推理时还在吗?**

- **训练手段**(EvoCUA / CoVe / Tool-R0):
  - verification / sandbox / synthetic data 都在**训练阶段**
  - 训练完丢掉,推理时单跑
  
- **Agent harness**(GEMS / CiteAudit):
  - memory / skills / multi-agent pipeline 在**推理时一直存在**
  - base model 不变,在它周围搭脚手架

**类比**: 
- 训练 base model = 让学生学会做题(学生本人变了)
- Agent harness = 学生不变,给他草稿纸 + 计算器(外挂工具)

**老师说的 "agent harness"(memory / verification / self-correction)** 
属于第二类 — 推理时的脚手架。

### Q4-上: GUI agent 的 Observation space

GUI agent "看"屏幕的两大流派:

**🅰️ 结构化派 (Structured / Symbolic)**
- DOM tree (web) / accessibility tree (mobile, desktop) / Set-of-Marks
- ✅ 精确、token 高效、坐标可靠
- ❌ 不通用(很多 GUI 没 a11y),不像人

**🅱️ 视觉派 (Visual / Pixel-based)**
- Screenshot only / screenshot + bounding boxes
- ✅ 最通用,跟人一致
- ❌ Visual grounding 是核心难题

**2026 年主流趋势**:视觉派(EvoCUA / Kimi K2.5 / UI-Venus 都是视觉为主)

**热点子方向**:Visual grounding(ScreenSpot benchmark 在测的)
- UI-Venus 在 ScreenSpot-Pro 上 69.6%
- 这个数字直接反映"agent 能不能精确点击 UI 元素"

### Q4-下: GUI agent 的 Action space

Agent 表达"我要做什么动作"的三种方式:

**🅰️ Element-based**: `click(element_id="submit_btn")`
- 配 🅰️ 结构化 observation(DOM/a11y)
- 精确、token 高效,但需要底层 API

**🅱️ Coordinate-based**: `click(x=340, y=812)`
- 配 🅱️ 视觉 observation(screenshot)
- 通用,跟人一致,但 grounding 难
- **2026 主流**: EvoCUA, UI-Venus, Kimi K2.5 都走这条

**🅲️ Code-based**: 输出 Python/JS 代码块
- 配混合 observation
- 表达力最强,但有 bug / 安全风险
- 代表: Code2World, SWE agent

### Key insight
Observation 和 Action 是**配对**的:agent 怎么"看",就怎么"动"。

走视觉派(🅱️🅱️),**最核心的难题是 visual grounding**。
- ScreenSpot 是测这个的 benchmark
- UI-Venus 69.6% 反映的就是这种能力
- 2026 年 GUI agent 卷的核心战场就是这个


## Day-end reflection
