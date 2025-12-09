《
# 任务目标：对当前项目进行全生命周期（训练+推理）的微观代码解析，并生成严格遵循数据流转顺序的单张统一逻辑架构图

## Phase B-Global（全流程融合微观架构图）

### 核心逻辑
你需要将任务分为"规划"与"执行"两个步骤。

**关键约束**：生成的 Mermaid 图必须是一张融合图，共用的前向传播路径只画一次，然后在末端分叉展示训练和推理的不同终点。

- **规划**：生成 `viewB/checklist.md`，梳理出"公共路径"、"训练独有路径"和"推理独有路径"。
- **执行**：基于拆解计划，生成 `viewB/view_full.md`。

---

## 1. 输出文件定义（双文件输出）

### 文件一：全流程拆解清单

**路径**：`viewB/checklist.md`

**内容要求**：
- 必须清晰区分哪些是共用的，哪些是独有的。

**格式要求（示例）**：

```markdown
# 全流程模块拆解清单 (基于真实代码路径)

## Part 1: 公共主干 (Shared Common Flow)
- [ ] 数据入口 (Dataset/Input)
- [ ] 模型主体 (Backbone/Neck/Encoder...)

## Part 2: 逻辑分叉点 (Branching Point)
- [ ] 在此处根据 mode (train/eval) 产生分支

## Part 3: 训练独有分支 (Training Branch)
- [ ] Loss 计算 (Loss Functions)
- [ ] 梯度/优化器 (Optimizer Step)

## Part 4: 推理独有分支 (Inference Branch)
- [ ] 后处理/解码 (Post-processing/Decoding)
- [ ] 结果可视化/保存 (Output)
```

### 文件二：微观逻辑全景图

**路径**：`viewB/view_full.md`

**内容**：仅包含 Mermaid 代码块。

**绘制逻辑（核心）**：
- **单图原则**：整张图必须包含完整的训练和推理过程。
- **共享节点**：共用的模块（如 Backbone）严禁重复绘制。
- **分支结构**：
  ```
  Input --> Shared Model --> Branching Point
  Branching Point --> Training Logic
  Branching Point --> Inference Logic
  ```

---

## 2. Mermaid 绘图规范（硬性标准）

### 2.1 节点结构规范

**矩形节点 []（功能/逻辑）**：
Label 必须包含 HTML 格式的 5 个具体要素，且必须左对齐：

```html
<div style="text-align:left">
1. <b>模块名</b>：{函数名/类名}<br/>
2. <b>功能</b>：{简述作用}<br/>
3. <b>处理</b>：{输入} --[操作]--> {输出}<br/>
4. <b>输入</b>：{变量名} {Shape} {物理含义}<br/>
5. <b>输出</b>：{变量名} {Shape} {物理含义}
</div>
```

**椭圆节点 (( ))（中间变换）**：
- **触发条件**：变量名变更、Reshape、Permute 或 逻辑分叉（Branch Split）时。
- **连线**：无箭头实线 `---`。

### 2.2 连线与流向规范

**数据流向 (Data Flow)**：
- 使用实线箭头 `-->`。
- **分支绘制**：在共用模块输出后，引出两条线：
  ```
  Shared_Output --> |Training Path| Loss_Calculation
  Shared_Output --> |Inference Path| Post_Process
  ```

**函数调用 (Function Call)**：
- 使用带箭头的虚线 `-.->>`，必须形成闭环。

### 2.3 视觉风格

**分组 (Subgraph)**：

建议结构：
```mermaid
subgraph Shared_Flow [公共主干]
subgraph Training_Branch [训练独有]
subgraph Inference_Branch [推理独有]
```

**配色 (ClassDef)**：三个部分必须使用明显不同的颜色方案以示区分。

### 2.4 底部图例（Legend）

必须在 Mermaid 代码的最后添加以下标准图例：

```mermaid
%% --- 图例说明区 ---
subgraph Legend [图例说明]
    direction LR
    L1[矩形] ---|代表| L_Text1[功能/逻辑模块]
    L2((椭圆)) ---|代表| L_Text2[中间变换/分叉点]
    L3[A] -->|实线| L4[B]
    L4 ---|代表| L_Text3[数据流向 (Data Flow)]
    L5[C] -.->|虚线| L6[D]
    L6 ---|代表| L_Text4[函数调用 (Function Call)]
end
```

---

## 3. 最终执行检查

在生成响应前，请自检：

- [ ] 是否为单张融合图？有没有把训练和推理画在同一张画布上？
- [ ] 是否避免了重复？共用的 Backbone 是否只画了一次？
- [ ] 分支是否清晰？是否一眼能看出从哪里开始分叉？
- [ ] 文件是否分离？Checklist 和 Mermaid 是否分属不同文件？
- [ ] 图例是否存在？

现在，请开始分析代码，构建这张融合了训练与推理的全景图。
》
