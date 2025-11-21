"""
多模态内容处理的提示词模板

包含模态处理器中用于分析不同类型内容（图像、表格、公式等）的所有提示词模板
"""

from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# 不同分析类型的系统提示词
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "你是一位专业的图像分析专家。请提供详细、准确的描述。"
)
PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "你是一位专业的图像分析专家。请基于可用信息提供详细分析。"
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "你是一位专业的数据分析专家。请提供详细的表格分析和具体见解。"
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "你是一位专业的数学专家。请提供详细的数学分析。"
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "你是一位专门研究 {content_type} 内容的专业内容分析师。"
)

# 图像分析提示词模板
PROMPTS[
    "vision_prompt"
] = """请详细分析此图像，并提供以下结构的 JSON 响应：

{{
    "detailed_description": "按照以下指南提供全面详细的图像视觉描述：
    - 描述整体构图和布局
    - 识别所有对象、人物、文本和视觉元素
    - 解释元素之间的关系
    - 注明颜色、光线和视觉风格
    - 描述显示的任何动作或活动
    - 如果相关，包含技术细节（图表、示意图等）
    - 始终使用具体名称而不是代词",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "图像内容及其重要性的简明摘要（最多 100 字）"
    }}
}}

额外上下文：
- 图像路径：{image_path}
- 标题：{captions}
- 脚注：{footnotes}

重点提供准确、详细的视觉分析，以便用于知识检索。"""

# 带上下文支持的图像分析提示词
PROMPTS[
    "vision_prompt_with_context"
] = """请详细分析此图像，同时考虑周围的上下文。提供以下结构的 JSON 响应：

{{
    "detailed_description": "按照以下指南提供全面详细的图像视觉描述：
    - 描述整体构图和布局
    - 识别所有对象、人物、文本和视觉元素
    - 解释元素之间的关系以及它们与周围上下文的关联
    - 注明颜色、光线和视觉风格
    - 描述显示的任何动作或活动
    - 如果相关，包含技术细节（图表、示意图等）
    - 在相关时引用与周围内容的联系
    - 始终使用具体名称而不是代词",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "图像内容、其重要性以及与周围内容关系的简明摘要（最多 100 字）"
    }}
}}

来自周围内容的上下文：
{context}

图像详情：
- 图像路径：{image_path}
- 标题：{captions}
- 脚注：{footnotes}

重点提供准确、详细的视觉分析，融入上下文，以便用于知识检索。"""

# 带文本回退的图像分析提示词
PROMPTS["text_prompt"] = """基于以下图像信息提供分析：

图像路径：{image_path}
标题：{captions}
脚注：{footnotes}

{vision_prompt}"""

# 表格分析提示词模板
PROMPTS[
    "table_prompt"
] = """请分析此表格内容，并提供以下结构的 JSON 响应：

{{
    "detailed_description": "对表格进行全面分析，包括：
    - 表格结构和组织方式
    - 列标题及其含义
    - 关键数据点和模式
    - 统计见解和趋势
    - 数据元素之间的关系
    - 所呈现数据的重要性
    始终使用具体的名称和值而不是一般性引用。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "表格用途和关键发现的简明摘要（最多 100 字）"
    }}
}}

表格信息：
图像路径：{table_img_path}
标题：{table_caption}
内容：{table_body}
脚注：{table_footnote}

重点从表格数据中提取有意义的见解和关系。"""

# 带上下文支持的表格分析提示词
PROMPTS[
    "table_prompt_with_context"
] = """请分析此表格内容，同时考虑周围的上下文，并提供以下结构的 JSON 响应：

{{
    "detailed_description": "对表格进行全面分析，包括：
    - 表格结构和组织方式
    - 列标题及其含义
    - 关键数据点和模式
    - 统计见解和趋势
    - 数据元素之间的关系
    - 所呈现数据与周围上下文的相关性
    - 表格如何支持或说明周围内容的概念
    始终使用具体的名称和值而不是一般性引用。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "表格用途、关键发现以及与周围内容关系的简明摘要（最多 100 字）"
    }}
}}

来自周围内容的上下文：
{context}

表格信息：
图像路径：{table_img_path}
标题：{table_caption}
内容：{table_body}
脚注：{table_footnote}

重点从表格数据中提取有意义的见解和关系，结合周围内容的上下文。"""

# 公式分析提示词模板
PROMPTS[
    "equation_prompt"
] = """请分析此数学公式，并提供以下结构的 JSON 响应：

{{
    "detailed_description": "对公式进行全面分析，包括：
    - 数学含义和解释
    - 变量及其定义
    - 使用的数学运算和函数
    - 应用领域和上下文
    - 物理或理论意义
    - 与其他数学概念的关系
    - 实际应用或用例
    始终使用具体的数学术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "公式用途和重要性的简明摘要（最多 100 字）"
    }}
}}

公式信息：
公式：{equation_text}
格式：{equation_format}

重点提供数学见解并解释公式的重要性。"""

# 带上下文支持的公式分析提示词
PROMPTS[
    "equation_prompt_with_context"
] = """请分析此数学公式，同时考虑周围的上下文，并提供以下结构的 JSON 响应：

{{
    "detailed_description": "对公式进行全面分析，包括：
    - 数学含义和解释
    - 变量及其在周围内容上下文中的定义
    - 使用的数学运算和函数
    - 基于周围材料的应用领域和上下文
    - 物理或理论意义
    - 与上下文中提到的其他数学概念的关系
    - 实际应用或用例
    - 公式如何与更广泛的讨论或框架相关联
    始终使用具体的数学术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "公式用途、重要性以及在周围上下文中作用的简明摘要（最多 100 字）"
    }}
}}

来自周围内容的上下文：
{context}

公式信息：
公式：{equation_text}
格式：{equation_format}

重点提供数学见解并解释公式在更广泛上下文中的重要性。"""

# 通用内容分析提示词模板
PROMPTS[
    "generic_prompt"
] = """请分析此 {content_type} 内容，并提供以下结构的 JSON 响应：

{{
    "detailed_description": "对内容进行全面分析，包括：
    - 内容结构和组织方式
    - 关键信息和元素
    - 组件之间的关系
    - 上下文和重要性
    - 与知识检索相关的详细信息
    始终使用适合 {content_type} 内容的具体术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "内容用途和要点的简明摘要（最多 100 字）"
    }}
}}

内容：{content}

重点提取对知识检索有用的有意义信息。"""

# 带上下文支持的通用内容分析提示词
PROMPTS[
    "generic_prompt_with_context"
] = """请分析此 {content_type} 内容，同时考虑周围的上下文，并提供以下结构的 JSON 响应：

{{
    "detailed_description": "对内容进行全面分析，包括：
    - 内容结构和组织方式
    - 关键信息和元素
    - 组件之间的关系
    - 与周围内容相关的上下文和重要性
    - 此内容如何连接或支持更广泛的讨论
    - 与知识检索相关的详细信息
    始终使用适合 {content_type} 内容的具体术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "内容用途、要点以及与周围上下文关系的简明摘要（最多 100 字）"
    }}
}}

来自周围内容的上下文：
{context}

内容：{content}

重点提取对知识检索有用的有意义信息，并理解内容在更广泛上下文中的作用。"""

# 模态块模板
PROMPTS["image_chunk"] = """
图像内容分析：
图像路径：{image_path}
标题：{captions}
脚注：{footnotes}

视觉分析：{enhanced_caption}"""

PROMPTS["table_chunk"] = """表格分析：
图像路径：{table_img_path}
标题：{table_caption}
结构：{table_body}
脚注：{table_footnote}

分析：{enhanced_caption}"""

PROMPTS["equation_chunk"] = """数学公式分析：
公式：{equation_text}
格式：{equation_format}

数学分析：{enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} 内容分析：
内容：{content}

分析：{enhanced_caption}"""

# 查询相关提示词
PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "请简要描述此图像中的主要内容、关键元素和重要信息。"
)

PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "你是一位能够准确描述图像内容的专业图像分析师。"
)

PROMPTS[
    "QUERY_TABLE_ANALYSIS"
] = """请分析以下表格数据的主要内容、结构和关键信息：

表格数据：
{table_data}

表格标题：{table_caption}

请简要总结表格的主要内容、数据特征和重要发现。"""

PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "你是一位能够准确分析表格数据的专业数据分析师。"
)

PROMPTS[
    "QUERY_EQUATION_ANALYSIS"
] = """请解释以下数学公式的含义和用途：

LaTeX 公式：{latex}
公式标题：{equation_caption}

请简要说明此公式的数学含义、应用场景和重要性。"""

PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "你是一位能够清楚解释数学公式的数学专家。"
)

PROMPTS[
    "QUERY_GENERIC_ANALYSIS"
] = """请分析以下 {content_type} 类型内容，并提取其主要信息和关键特征：

内容：{content_str}

请简要总结此内容的主要特征和重要信息。"""

PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "你是一位能够准确分析 {content_type} 类型内容的专业内容分析师。"
)

PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\n请基于用户查询和提供的多模态内容信息提供全面的答案。"
)
