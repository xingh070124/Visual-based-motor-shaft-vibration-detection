# Paper: Perspective-Corrected Monocular Vision for Motor Shaft Vibration Measurement

> 目标期刊：IEEE Transactions on Instrumentation and Measurement (IEEE TIM)
> 模板：IEEEtran.cls (v1.8b)
> 状态：草稿完成，已编译通过

## 文件说明

| 文件 | 说明 |
|------|------|
| `manuscript.tex` | 论文主文件 |
| `references.bib` | BibTeX 参考文献 |
| `cover_letter.tex` | 投稿 Cover Letter |
| `IEEEtran.cls` | IEEE 期刊文档类 |
| `IEEEtran.bst` | IEEE 参考文献格式 |
| `manuscript.pdf` | 编译输出 (6 页, ~265KB) |
| `figures/` | 论文插图 |

## 编译

```bash
cd paper
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## 论文结构

1. **Abstract** — 方法总结 + 关键结果
2. **Introduction** — 工业背景 + 贡献列表
3. **Related Work** — 视觉振动测量 / 目标跟踪 / 椭圆检测
4. **Methodology** — 系统流水线 → 相机模型 → 轴检测 → STARK 跟踪 → **透视修正 scale**（核心创新）→ 振动分析
5. **Experiments** — 640 张仿真图像 + 10-100 RPM 真实电机
6. **Results & Discussion** — 倾角精度 / 振幅误差 / 误差来源分解 / 噪声模糊分析 / 真实视频结果
7. **Conclusion** — 总结 + 局限 + 未来工作

## 核心公式

**透视修正各向同性比例尺**：

$$s = \frac{D \cdot a}{2b^2}$$

其中 $D$ = 轴直径 (12mm), $a,b$ = 椭圆半长轴/半短轴 (像素)

**倾角反演**：$\theta = \arccos(b/a)$

## 关键数据

| 指标 | 值 |
|------|-----|
| 倾角估计精度 (θ≥3°) | < 1° |
| Scale 公式误差 | < 0.1% |
| 振幅误差主因 | fitEllipse 中心精度 (~0.1px) |
| 真实视频 scale 稳定性改善 | 2.9× |
| 漂移减少 | X: 3×, Y: 2× |

## 待完成事项

- [ ] 填入真实大学名称和作者邮箱
- [ ] 补充系统流水线示意图 (Fig. 1)
- [ ] 补充透视投影几何示意图 (Fig. 2)
- [ ] 补充真实视频振动幅度-RPM 图 (Fig. 6)
- [ ] 补充激光传感器对比数据
- [ ] 最终语言润色
- [ ] 同行预审
