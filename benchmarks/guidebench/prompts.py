"""Verbatim MIT-licensed prompts from Dlxxx/GuideBench, commit 78c5bfa42facee34db31e4ba03ad2c3b5a04bbbc."""

ANSWER_TEMPLATE = """你的任务是根据提供的指令、文案以及候选选项的内容，选出最佳选项并给出简要分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    候选选项：
    <Options>{MultipleOptions}</Options>
    评估选项时，需要注意：
    1. 仔细比对每个选项与指令<Instruction>和<Guidelines>的符合程度、准确性与完整性。
    输出json格式。包括两个字段，OptimalOption和AnswerAnalysis。其中OptimalOption仅输出最佳选项序号，AnswerAnalysis中给出简要分析，不要指出rule_id和type，简要比较最佳选项和其他选项的区别。
"""

QA_TEMPLATE = """你的任务是根据提供的指令、文案输出判断结果与具体分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    输出json格式。包括两个字段，CandidateAnswer和CandidateAnalysis。其中CandidateAnswer仅输出0或1，0代表否定，1代表肯定；CandidateAnalysis中给出简要分析，不要指出rule_id和type。
"""

MATH_QA_TEMPLATE = """你的任务是根据提供的指令、文案输出判断结果与具体分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    输出json格式。包括两个字段，CandidateAnswer和CandidateAnalysis。其中CandidateAnswer仅输出计算结果，字符串格式，不加单位“元”；CandidateAnalysis中给出必要的计算过程，字符串格式，不要指出rule_id和type。
"""

RE_QA_TEMPLATE = """你的任务是根据提供的指令、文案输出判断结果与具体分析。
    首先，请仔细理解：
    Instruction为通用任务目标，内容为：
        <Instruction>{Instruction}</Instruction>
    Guideline为用户定义的规则，内容为：
        <Guidelines>{Guidelines}</Guidelines>
    文案内容：
    <Context>{Context}</Context>
    输出json格式。包括两个字段，CandidateAnswer和CandidateAnalysis。其中CandidateAnswer仅输出以下三种判断结果之一，"2（强相关）","1（弱相关）","0（不相关）"；CandidateAnalysis中给出必要分析过程，字符串格式，不要指出rule_id和type。
"""
