from pathlib import Path
import pandas as pd,numpy as np,json,zipfile
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/modeling/retrospective'
OUT = ROOT / 'analysis_outputs/reproduction/retrospective'
OUT.mkdir(parents=True, exist_ok=True)
r = pd.read_csv(DATA / 'all_predictions.csv')
case=r.pivot(index=['enzyme','group','status'],columns='target',values='fold_error').reset_index()
case.to_csv(OUT/'parameter_error_matrix.csv',index=False,encoding='utf-8-sig')
primary=r[r.status.eq('primary') & ~r.identical_training_sequence]
summary=primary.groupby('target').agg(n=('enzyme','size'),MAE_log10=('absolute_log10_error','mean'),baseline_MAE_log10=('baseline_absolute_log10_error','mean'),median_fold_error=('fold_error','median'))
summary.to_csv(OUT/'primary_all_target_summary.csv',encoding='utf-8-sig')
complete=primary[primary.target.isin(['Km_uM','kcat_s'])].groupby('enzyme').filter(lambda x:len(x)==2)
none_both_three=complete.groupby('enzyme').fold_error.max().le(3).sum()
assert none_both_three==0
def table(names,targets):
 data=r[r.enzyme.isin(names)&r.target.isin(targets)]
 lines=['| 酶 | 参数 | 实测值 | 预测值 | 误差倍数 | 均值基线误差倍数 |','|---|---|---:|---:|---:|---:|']
 for x in data.itertuples():lines.append(f'| {x.enzyme} | {x.target} | {x.observed:.6g} | {x.predicted:.6g} | {x.fold_error:.3g} | {x.baseline_fold_error:.3g} |')
 return '\n'.join(lines)
groups=pd.read_csv(DATA / 'whole_study_rankings.csv')
report='''# 扩大检索后的新酶动力学预测核查

## 结论

找到了比此前 DON 案例更接近的单参数预测。kcat 最接近的两个新增案例为 ZenA_Sl 和 ZenA_Scfl，误差分别为 1.22 和 1.88 倍；同一篇研究中的 ZHDAY3 WT 和 N153H 的 Km 误差分别为 1.13 和 1.05 倍。但尚未找到 kcat、Km 同时稳定预测较好的完整案例组。可核验的双参数候选中，没有一个案例的这两个参数误差同时在 3 倍以内；这里的 3 倍只是描述结果，不是预注册成功标准。

这些是看过扩大后的结果后选出的最佳例子，不能作为无偏外部性能评估，也不能证明模型发现了新降解酶。原答复信和固定正文/SI没有因这轮探索而修改。

## 检索与模型

- 检索日期为 2026 年 10 月 25 日，重点覆盖 2025–2026 年，并扩大至 2023–2024 年没有进入原建模训练集的研究；并非穷尽的系统综述。
- 筛选记录列出 24 项研究，分别标明全文、补充材料或仅摘要核实范围。部分文章因全文/SI无法获取而待核对，不能解释成不存在相关数据。
- 共计算 22 个酶/变体、50 项已报告参数。其中 19 个候选、43 项参数属于主分析映射；其余 MgSDR1、ZENM 和 AsPLa-D 共 3 个候选、7 项参数因序列代理或构建体问题单列为探索性结果。
- 使用相同原训练数据、冻结 ESM-2 序列表示、底物特征及 StandardScaler + Ridge α=10；未根据新案例换模型、调参或修改原训练数据。该结论针对固定 Ridge 配置，不代表穷尽测试了所有候选模型。
- 所有测试过的结果都保留，包括此前的 DONepi、DONrd。所有候选均没有完全相同的训练序列，但部分同源酶仍接近训练家族，不能称为全部都是未见家族。
- 误差倍数为 max(预测值/实测值, 实测值/预测值)，1 倍为完全一致。kcat 单位 s⁻¹，Km 单位 μM，kcat/Km 单位 M⁻¹s⁻¹。不能把误差倍数转换成所谓总体准确率。

## 一 最接近的 kcat 案例

来源：[Fruhauf 等 ACS Catalysis 2024](https://doi.org/10.1021/acscatal.4c00271)，原文 Table 1。底物为玉米赤霉烯酮（ZEN）。ZenA_Sl 来自 Streptomyces lividans，ZenA_Scfl 来自 Streptomyces coelicoflavus。它们是原文 9 个有测定值的 ZenA 同源酶中的两个，不是整篇研究都预测良好。

'''+table(['ZenA_Sl','ZenA_Scfl'],['kcat_s','Km_uM'])+'''

其中 ZenA_Sl 的 kcat 优于训练均值基线；ZenA_Scfl 的 kcat 虽在约 2 倍误差以内，训练均值基线反而更接近。二者的 Km 误差分别达到约 13 和 36 倍。因此只能作为 selected kcat examples 展示，不能写成这两个酶的全部动力学参数都预测准确。

整篇 ZenA 研究的全部 9 个酶、18 项测量（kcat 与 Km 两个目标各 9 项）的平均绝对 log10 误差为 1.243，对应几何误差因子 17.51；均值基线为 22.51。已有全部结果没有因挑选两例而被删除。

## 二 Km 最接近的完整双酶组

来源：[Wang 等 Applied and Environmental Microbiology 2024](https://doi.org/10.1128/aem.01818-23)，原文 Table 3，底物为 α-zearalanol（α-ZAL），不是 ZEN。原文明确给出 XP_013255140.1，N153H 由该序列按位点重建，并核对原位点为 N。

'''+table(['ZHDAY3_WT','ZHDAY3_N153H'],['Km_uM','kcat_s','kcat_Km_M_s'])+'''

这个双酶组的 Km 几何误差因子约为 1.09，均值基线约为 1.17；但 kcat 误差约为 28 和 35 倍，效率误差约为 18 和 26 倍。模型将两种 Km 预测得很接近，对突变带来的变化幅度刻画也有限。不能把较好的 Km 结果推广为三个目标都可靠。

## 三 以整项研究比较时

共同指标只使用同时具备 kcat 与 Km 的候选，计算两个目标的平均绝对 log10 误差，不能把各案例最好的那个参数混在一起。至少 3 个有双参数的酶/变体才作为多酶组比较。

- ZenA 的 9 个酶构成满足该条件的较好研究组：几何误差因子 17.51 倍，仍不足以称为整体预测准确。
- ZHRnZ 的 WT、E122Q、E122R 组为 25.13 倍。
- ZHDAY3 双酶组只有 2 个酶，其双目标几何误差因子为 5.84 倍，均值基线为 2.74 倍，整体并不优于基线。

## 四 其他值得注意的单参数结果

- ZENM 的 Km 报告为 21.59 μM，预测为 24.35 μM，误差 1.13 倍。但目前以早期文献的参考序列映射，尚未从该项 2026 研究全文确认实际表达构建体，故不进入严格最佳案例排名。
- MgSDR1 的 kcat/Km 报告为 8.66 M⁻¹s⁻¹，预测为 9.99 M⁻¹s⁻¹，误差 1.15 倍。但其 kcat 和 Km 分别相差约 28 和 12 倍，且原图 accession 疑似 NP/XP 排印错误，已单列序列映射问题。
- 2026 年 AsPLa-D 使用作者 SI 的设计序列计算，而不是误用 WT。kcat、Km 和效率误差分别约 3.08、8.90、16.80 倍；实验采用 MBP 融合表达，构建体效应是额外限制。

## 五 对答复信的建议

可以展示有利的单参数例子，但必须同时提供全部测试结果，明确是扩大检索后的事后选例。不能用两例成功参数替换其他失败结果，写成模型已经能成功发现新酶。

如需加入补充回复，可用以下简短表述（该段未自动写入答复信）：

> We expanded the retrospective check to additional published enzyme–mycotoxin measurements using the same fixed Ridge configuration and retained all evaluated cases. Selected kcat predictions were within twofold of the reported values, and Km estimates were close for the ZHDAY3 wild type and N153H variant. Performance nevertheless varied substantially across enzymes and targets, and these examples were identified after evaluation. They therefore do not establish prospective enzyme discovery or consistently improved prediction over the training-mean baseline.

## 文件与核查

- `all_predictions.csv`：全部 50 项结果及逐项基线、序列匹配信息。
- `parameter_error_matrix.csv`：同一酶三个目标的误差并列，避免只看有利目标。
- `complete_case_rankings.csv`：双参数联合误差排名；须按 status 区分代理/构建体候选。
- `whole_study_rankings.csv`：整项研究的双参数结果，不能把 n=1 的条目称为多酶组。
- `primary_all_target_summary.csv`：严格映射候选的各目标整体结果。
- `screening_ledger.csv`：已检查的研究及纳入、待核实、未预测原因。
- `candidate_provenance.json`：输入序列、原文参数、单位转换和限制。
- `analysis_protocol.md`：扩大预测执行前确定的固定配置、完整案例及整组排名规则。

输入、训练数据及特征文件的哈希已核对，原文件未改变。全部输出是回顾性计算，没有新增湿实验，也没有修改固定正文或 SI。
'''
(OUT/'扩大检索与预测结果.md').write_text(report,encoding='utf8')
print(summary.to_string());print('Primary candidates',primary.enzyme.nunique(),'instances',len(primary),'No complete case within 3-fold on both targets:',none_both_three==0)
