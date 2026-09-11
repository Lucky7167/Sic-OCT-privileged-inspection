# SiC 晶圆 LUPI Teacher–Student 训练代码包

与方法学文档（晶圆缺陷定位 new(4)）逐节对应的完整训练代码。

## 文件清单与训练顺序

| 顺序 | 脚本 | 对应文档 | 输入 | 输出 |
|---|---|---|---|---|
| 0 | （离线）DINOv2 特征预提取 | 2.2/2.3 | M_local / M_ctx 图像 | `data/patches/*.npy` |
| 1 | `train_mae.py` | 第4章 阶段A | 无标签 OCT 体数据 | `checkpoints/mae_r3d18.pt` |
| 2 | `train_aux_classifier.py` | 3.6 | [CLS] 特征 + z* | `aux_classifier.pt` + 回填 r_i/R/U* |
| 3 | `train_teacher.py` | 第5章 阶段B | 四模态 + 全部真值 | `teacher.pt` |
| 4 | `save_lessons.py` | 第6章 阶段C | teacher.pt（冻结） | `lessons.pt` |
| 5a | `train_direct.py` | 7.8 | 三件套（无教案） | `direct.pt` |
| 5b | `train_student.py` | 第7章 阶段D | 三件套 + 教案 | `student.pt`（部署模型） |
| 6 | `evaluate.py` | 7.6/7.8 | 三个 ckpt | figures/ 五张图 + 指标报告 |

5a 与 5b 互不依赖，可并行。支撑文件：`common.py`（配置/数据/损失/指标）、`student_model.py`（Student 结构，Direct 共用）。

## 数据准备（训练前）

```
data/
├── manifest.csv      # 每行一个 ROI：roi_id, wafer, split, x,y,w,h,a,
│                     #   z_star, q_star, d,v,o,c(NaN), U_star(先留空)
├── patches/{roi_id}_local.npy   # DINOv2 [CLS] (384,)，离线提取一次
├── patches/{roi_id}_ctx.npy
└── volumes/{roi_id}.npy         # OCT V (3,64,64,64) float32
```

- split 划分：Wafer A/B 内部 8:2 切 train/val；C 整片 val；D/E 整片 test。**按晶圆分，绝不按 ROI 随机分**（防组泄漏）。
- 归一化常数由 `common.compute_or_load_norm` 只在 A/B 训练部分计算并冻结到 `norm_constants.json`。
- `U_star` 先留空，跑完 `train_aux_classifier.py` 自动回填（含 m_bar）。

## 评估指标速查

**证明 Direct 预测不了深度缺陷（论文叙事第一幕）的证据链：**
1. **subsurface 召回率**（recall[2]）——Direct 应显著低于 Student
2. **混淆矩阵 conf[2,1]**——subsurface→surface 误判率（深度歧义直接证据）
3. **subsurface 组内 Spearman**——只对 z*=2 样本算 U 排序
4. paired bootstrap ΔSpearman 95% CI 不含 0（统计显著性）

不要用准确率当主指标（类别不平衡下有误导性）。

各网络自身指标：MAE 看重建 MSE 趋势；辅助分类器看 macro-F1（够用即可）；
Teacher 看验证集 L_U + Spearman（早停）；Student 看 7.6 验收三条
（Spearman 差距≤0.05、准确率差距≤3%、CPU<10ms）。

## 运行

```bash
pip install torch torchvision pandas numpy matplotlib
python train_mae.py
python train_aux_classifier.py
python train_teacher.py
python save_lessons.py
python train_direct.py & python train_student.py & wait
python evaluate.py        # 默认在 Wafer C 上评估；D/E 只开封一次
```

## 注意

- 代码未经运行，首次跑通时先小数据量 smoke test（把 epoch 改 2）。
- LOWO（第9章）：复制 `common.CFG` 中 TRAIN/VAL/TEST_WAFERS 为每折配置，重跑 1–5b 即可，无需新代码。
- DINOv2 特征预提取脚本未包含：用 `torch.hub.load('facebookresearch/dinov2','dinov2_vits14')` 对 resize 到 3×224×224 的 patch 前向取 [CLS] 即可，三网络复用。
