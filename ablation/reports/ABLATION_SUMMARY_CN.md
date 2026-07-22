# VR-TriRef 模态消融实验总结

本文档总结当前 `ablation/` 下已经完成的三个实验的描述性消融结果。

结果来源以以下两个全量 summary 为准：

- 实验一：`ablation/exam1/reports/exam1_ablation_summary.csv`
- 实验二：`ablation/exam2/reports/exam2_ablation_summary.csv`
- 实验三：`ablation/exam3/reports/full_v3/ablation_summary.csv`

注意：`ablation/reports/ABLATION_RESULTS.md` 中实验一 `no_gaze`、`no_hand` 曾出现过 2 条样本的旧 smoke 行；本文档采用实验一单独报告中的 4000 条全量结果。

## 实验设置

### 实验一：Closed-set 3D anchor selection

任务：模型从 scene-level candidate anchors 中选择用户指代对象。

Baseline：

- 路径：`data/match_eval_qwen3vl30b_mention_first_v3/`
- 样本数：4000
- Candidate anchor list、parser、evaluator 保持不变。

消融条件：

- `language_anchors_only`：保留语言和 candidate anchors，去掉主要多模态证据。
- `no_visual`：不传原始视频，改用空白 `64x64` RGB placeholder，属于较干净的视觉移除控制。
- `no_gaze`：移除结构化 gaze 字段、gaze summary、gaze-derived timeline proposals；但原始视频仍可能包含可见 gaze marker。
- `no_hand`：移除结构化 hand summary、手部/ray 相关坐标字段；但视频里的可见手部动作仍存在。

### 实验二：Projected-2D point diagnostic

任务：将 GT 3D anchors 投影到第一人称 image panels 上，评估 temporal evidence selection、point@K、joint@K。

Baseline：

- 路径：`exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`
- 事件数：4000
- GT referents：8330

消融条件：

- `full_panels_no_crop`：去掉 gaze-centered crop，只保留 full panels；这是 crop/path ablation，不是完整视觉消融。
- `instruction_only_prompt`：改成 instruction-only prompt；这是 prompt-format ablation，不是纯模态消融。
- `no_gaze_text_prior`：去掉 gaze prompt/text prior，但不遮挡图像中的 gaze marker。
- `no_gaze`：去掉 gaze prompt/text prior，并 mask 掉 panel 图像里的 projected green gaze marker，是实验二当前最干净的 gaze 消融。

实验二当前 manifest/prompt 没有显式 hand summary 或 hand joints 接口，因此不应声称实验二完成了 hand ablation。

## 实验一结果

| Variant | Predictions | Overall Acc | Delta Acc | Mapped Acc | Exact Set Acc | Micro F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_baseline | 4000 | 0.7660 | - | 0.7714 | 0.3205 | 0.5326 | - |
| language_anchors_only | 4000 | 0.6280 | -0.1380 | 0.6324 | 0.2553 | 0.4896 | -0.0430 |
| no_gaze | 4000 | 0.7170 | -0.0490 | 0.7221 | 0.2840 | 0.4890 | -0.0436 |
| no_hand | 4000 | 0.7665 | +0.0005 | 0.7719 | 0.3180 | 0.5326 | 0.0000 |
| no_visual | 4000 | 0.7302 | -0.0358 | 0.7354 | 0.3147 | 0.5425 | +0.0099 |

### 实验一结论

1. `language_anchors_only` 降幅最大：overall accuracy 从 `0.7660` 降到 `0.6280`，下降 `13.80` 个百分点。这说明视觉、gaze、hand 等多模态证据整体对 closed-set anchor selection 有贡献。

2. `no_gaze` 有明确下降：overall accuracy 下降 `4.90` 个百分点，micro F1 下降 `4.36` 个百分点。这支持 gaze structured cue 对实验一有帮助。

3. `no_visual` overall accuracy 下降 `3.58` 个百分点，但 micro F1 从 `0.5326` 小幅升到 `0.5425`。因此不能写成“所有指标均下降”。更稳妥的说法是：视觉移除降低了 overall closed-set selection accuracy，但 set-level F1 没有同步下降。

4. `no_hand` 基本无变化：overall accuracy `+0.0005`，micro F1 `0.0000`。当前实验一结果不支持“hand cue 明显提升 closed-set selection”的结论。

## 实验二结果

| Variant | Events | Predictions | Time F1 | Point@100 F1 | Joint@100 F1 | Delta Joint@100 | Mean Point Dist@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_baseline | 4000 | 8635 | 0.7333 | 0.2470 | 0.2038 | - | 57.9929 |
| full_panels_no_crop | 4000 | 8637 | 0.7010 | 0.2236 | 0.1759 | -0.0280 | 58.7615 |
| instruction_only_prompt | 4000 | 7582 | 0.6549 | 0.2262 | 0.1799 | -0.0240 | 57.5738 |
| no_gaze | 4000 | 8632 | 0.6886 | 0.2027 | 0.1551 | -0.0488 | 59.0221 |
| no_gaze_text_prior | 4000 | 8617 | 0.6896 | 0.2199 | 0.1725 | -0.0313 | 58.2964 |

### 实验二结论

1. `no_gaze` 是当前最有力的 gaze contribution 证据。它同时移除 gaze text prior 和图像中的 projected gaze marker，Joint@100 F1 从 `0.2038` 降到 `0.1551`，下降 `4.88` 个百分点；Point@100 F1 从 `0.2470` 降到 `0.2027`，下降 `4.43` 个百分点。

2. `no_gaze_text_prior` 也下降，但幅度较小：Joint@100 F1 下降 `3.13` 个百分点，Point@100 F1 下降 `2.71` 个百分点。它没有遮挡可见 gaze marker，因此可解释为“去掉 gaze 文本/先验提示会伤害性能，但图像中的 gaze marker 仍能提供部分信息”。

3. `full_panels_no_crop` 下降 `2.80` 个百分点，说明 gaze-centered crop 路径对 temporal/point joint grounding 有帮助。

4. `instruction_only_prompt` 下降 `2.40` 个百分点，但这是 prompt-format ablation，不能直接作为某个模态贡献的证据。

## 实现审计结论

当前消融总体是可用的，但论文中需要准确表述边界。

可强声明：

- 实验二 gaze 消融是当前最干净、最有说服力的模态贡献证据。
- 实验一 gaze structured cue 移除后性能下降。
- 实验一 language+anchors-only 明显低于 full，说明多模态证据整体有价值。
- 实验二 crop/context 设计对 projected-2D diagnostic 有帮助。

应谨慎声明：

- 实验一 `no_gaze` 只移除了结构化 gaze 信息，没有像实验二那样做视频级 gaze marker masking。
- 实验一 `no_hand` 只移除了结构化 hand/ray 信息，视频里的可见手部动作仍然保留。
- 实验一 `no_visual` 的 overall accuracy 下降，但 micro F1 没有下降。
- 实验二不能声称 hand ablation，因为当前流程没有显式 hand modality 输入。

不建议声明：

- “所有模态移除都会导致所有指标下降。”
- “hand modality 在当前两个实验中有稳定正贡献。”
- “实验一 no_gaze/no_hand 是像素级完全移除。”
- “实验二 no_visual 已完成全量结果。”当前全量表里没有 blank visual 的实验二结果。

## Parser 和 invalid output 情况

实验一：

- Baseline full：`response_status_ok_count = 3970 / 4000`
- `no_visual`：`3906 / 4000`
- `no_gaze`：`3991 / 4000`
- `no_hand`：`3980 / 4000`
- `language_anchors_only`：`3947 / 4000`

`no_visual` invalid/non-ok 数量相对更高，但没有出现单一 scene 的灾难性格式崩溃。

实验二：

- `full_panels_no_crop`：0 parse failures / 3971 prediction groups
- `no_gaze`：0 parse failures / 3971 prediction groups
- `no_gaze_text_prior`：0 parse failures / 3971 prediction groups
- `instruction_only_prompt`：7 parse failures / 3971 prediction groups

实验二主要结果不是由 parser failure 驱动的。

## 论文写法建议

可以写成：

> We conduct modality ablations on the two established diagnostics. For closed-set 3D anchor selection, removing structured gaze cues reduces overall accuracy from 76.60% to 71.70%, while replacing the visual stream with a blank placeholder reduces it to 73.02%. A language+anchor-only control further drops performance to 62.80%, indicating that multimodal evidence contributes beyond the candidate-anchor prior. For the projected-2D diagnostic, removing both gaze textual priors and visible projected gaze markers yields the largest degradation, reducing Joint@100 F1 from 20.38% to 15.51%.

需要补一句限制：

> The experiment-1 gaze and hand ablations remove structured cue fields but do not perform pixel-level removal from the original videos; therefore they are reported as structured-cue ablations. Experiment 2 currently supports a clean gaze ablation but not a hand ablation because hand telemetry is not part of the evaluated 2D manifest interface.

## 后续如果要进一步加强

1. 对实验一做视频帧级 gaze marker masking，得到更严格的 `no_gaze`。
2. 对实验一做 hand region masking，或者改用 selected evidence frames 而不是原视频，以便做像素级 hand ablation。
3. 对实验二补一个 full blank-visual run，作为更彻底的 visual ablation。
4. 实验三应进一步设计在 frame selection 之前移除模态的严格对照，以区分 selector 依赖与模型推理依赖。

## 实验三结果

实验三采用冻结的 v9 `candidate-free measured point-hypothesis diagnostic`。五个变体使用相同的 4,000 条样本、evidence frame、Qwen3-VL-30B checkpoint、parser、greedy decoding 和 evaluator，仅在 panel 选择完成后遮蔽模型可见字段。

| Variant | Valid | Anchor F1 | 相对 Full | Exact | Margin-F1@1.0 | Margin-F1@2.0 | Scene-normalized error |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 4000 | 0.4326 | - | 0.0185 | 0.2503 | 0.4105 | 0.1569 |
| no_visual | 4000 | 0.4341 | +0.0014 | 0.0180 | 0.2518 | 0.4120 | 0.1560 |
| no_gaze | 4000 | 0.0673 | -0.3653 | 0.0013 | 0.0055 | 0.0263 | 0.3771 |
| no_hand | 4000 | 0.4353 | +0.0027 | 0.0187 | 0.2527 | 0.4134 | 0.1558 |
| no_hand_strict | 4000 | 0.4351 | +0.0025 | 0.0187 | 0.2525 | 0.4132 | 0.1558 |
| no_gaze_hand | 3999 | 0.0542 | -0.3784 | 0.0000 | 0.0000 | 0.0037 | 0.5376 |
| no_instruction | 4000 | 0.4332 | +0.0006 | 0.0177 | 0.2510 | 0.4111 | 0.1565 |

### 实验三结论与边界

1. 移除 gaze 后 Anchor F1 从 `0.4326` 降至 `0.0673`，Margin-F1@1.0 从 `0.2503` 降至 `0.0055`，表明当前 v9 协议强依赖模型可见的 gaze point hypotheses。

2. `no_visual`、`no_hand`、`no_hand_strict` 和 `no_instruction` 与 full 几乎一致；其中严格移除 hand telemetry 并遮盖可见手势后的 `no_hand_strict` Anchor F1 为 `0.4351`，仍有 `96.92%` 的样本输出与 Full 完全相同。这不证明 hand 在一般指代任务中无用，而说明当前 prompt 默认复制 gaze hypotheses，使其主导模型输出。

3. `no_hand_strict` 对 11,001 个 panel 全部生成独立输入图，其中 9,323 个 panel 的画内手部被中性灰遮盖，1,678 个 panel 的 tracked hand 投影在画外；4,000 条输出全部有效。它是严格的模型输入级 hand 消融，但冻结的 target-free frame selector 在遮蔽前使用了 hand availability 与 stability，因此仍不能写成整个 pipeline 的严格单模态因果消融。

4. `no_gaze_hand` 有一条 invalid：`scene4_room1::32` 输出了六维 point。该样本作为空预测保留在 4,000 分母中，未人工修改模型输出。

完整报告见 `ablation/exam3/reports/full_v3/EXPERIMENT3_QWEN30B_ABLATION.md` 和 `ablation/exam3/reports/strict_hand_v1/EXPERIMENT3_QWEN30B_ABLATION.md`；严格手部消融逐样本证据见 `paper_experiment_evidence/ablation/experiment3_qwen30b_strict_hand/`。
