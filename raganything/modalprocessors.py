"""
不同模态的专用处理器

包括：
- ContextExtractor: 多模态内容的通用上下文提取器
- ImageModalProcessor: 图像内容的专用处理器
- TableModalProcessor: 表格内容的专用处理器
- EquationModalProcessor: 公式内容的专用处理器
- GenericModalProcessor: 其他模态内容的处理器
"""

import re
import json
import time
import base64
from typing import Dict, Any, Tuple, List
from pathlib import Path
from dataclasses import dataclass

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
from lightrag.lightrag import LightRAG
from dataclasses import asdict
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
from lightrag.operate import extract_entities, merge_nodes_and_edges

# 导入提示词模板
from raganything.prompt import PROMPTS


@dataclass
class ContextConfig:
    """上下文提取的配置"""

    context_window: int = 1  # 上下文提取的窗口大小
    context_mode: str = "page"  # "page", "chunk", "token"
    max_context_tokens: int = 2000  # 最大上下文 token 数
    include_headers: bool = True  # 是否包含标题/标头
    include_captions: bool = True  # 是否包含图像/表格说明
    filter_content_types: List[str] = None  # 要包含的内容类型

    def __post_init__(self):
        if self.filter_content_types is None:
            self.filter_content_types = ["text"]


class ContextExtractor:
    """支持多种内容源格式的通用上下文提取器"""

    def __init__(self, config: ContextConfig = None, tokenizer=None):
        """初始化上下文提取器

        Args:
            config: 上下文提取配置
            tokenizer: 用于精确 token 计数的分词器
        """
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer

    def extract_context(
        self,
        content_source: Any,
        current_item_info: Dict[str, Any],
        content_format: str = "auto",
    ) -> str:
        """从内容源中提取当前项目的上下文

        Args:
            content_source: 源内容（列表、字典或其他格式）
            current_item_info: 当前项目的信息（page_idx、index 等）
            content_format: 内容源的格式提示（"minerU"、"text_chunks"、"auto" 等）

        Returns:
            提取的上下文文本
        """
        if not content_source and not self.config.context_window:
            return ""

        try:
            # 如果提供了格式提示则使用，否则自动检测
            if content_format == "minerU" and isinstance(content_source, list):
                return self._extract_from_content_list(
                    content_source, current_item_info
                )
            elif content_format == "text_chunks" and isinstance(content_source, list):
                return self._extract_from_text_chunks(content_source, current_item_info)
            elif content_format == "text" and isinstance(content_source, str):
                return self._extract_from_text_source(content_source, current_item_info)
            else:
                # 自动检测内容源格式
                if isinstance(content_source, list):
                    return self._extract_from_content_list(
                        content_source, current_item_info
                    )
                elif isinstance(content_source, dict):
                    return self._extract_from_dict_source(
                        content_source, current_item_info
                    )
                elif isinstance(content_source, str):
                    return self._extract_from_text_source(
                        content_source, current_item_info
                    )
                else:
                    logger.warning(
                        f"Unsupported content source type: {type(content_source)}"
                    )
                    return ""
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return ""

    def _extract_from_content_list(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """从 MinerU 样式的内容列表中提取上下文

        Args:
            content_list: 包含 page_idx 和类型信息的内容项列表
            current_item_info: 当前项目信息

        Returns:
            来自周围页面/块的上下文文本
        """
        if self.config.context_mode == "page":
            return self._extract_page_context(content_list, current_item_info)
        elif self.config.context_mode == "chunk":
            return self._extract_chunk_context(content_list, current_item_info)
        else:
            return self._extract_page_context(content_list, current_item_info)

    def _extract_page_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """基于页面边界提取上下文

        Args:
            content_list: 内容项列表
            current_item_info: 包含 page_idx 的当前项目

        Returns:
            来自周围页面的上下文文本
        """
        current_page = current_item_info.get("page_idx", 0)
        window_size = self.config.context_window

        start_page = max(0, current_page - window_size)
        end_page = current_page + window_size + 1

        context_texts = []

        for item in content_list:
            item_page = item.get("page_idx", 0)
            item_type = item.get("type", "")

            # 检查项目是否在上下文窗口内并符合过滤条件
            if (
                start_page <= item_page < end_page
                and item_type in self.config.filter_content_types
            ):
                text_content = self._extract_text_from_item(item)
                if text_content and text_content.strip():
                    # 添加页面标记以便更好地理解上下文
                    if item_page != current_page:
                        context_texts.append(f"[Page {item_page}] {text_content}")
                    else:
                        context_texts.append(text_content)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_chunk_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """基于内容块提取上下文

        Args:
            content_list: 内容项列表
            current_item_info: 包含索引信息的当前项目

        Returns:
            来自周围块的上下文文本
        """
        current_index = current_item_info.get("index", 0)
        window_size = self.config.context_window

        start_idx = max(0, current_index - window_size)
        end_idx = min(len(content_list), current_index + window_size + 1)

        context_texts = []

        for i in range(start_idx, end_idx):
            if i != current_index:
                item = content_list[i]
                item_type = item.get("type", "")

                if item_type in self.config.filter_content_types:
                    text_content = self._extract_text_from_item(item)
                    if text_content and text_content.strip():
                        context_texts.append(text_content)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_text_from_item(self, item: Dict) -> str:
        """从内容项中提取文本内容

        Args:
            item: 内容项字典

        Returns:
            提取的文本内容
        """
        item_type = item.get("type", "")

        if item_type == "text":
            text = item.get("text", "")
            text_level = item.get("text_level", 0)

            # 为结构化内容添加标题指示
            if self.config.include_headers and text_level > 0:
                return f"{'#' * text_level} {text}"
            return text

        elif item_type == "image" and self.config.include_captions:
            captions = item.get("image_caption", item.get("img_caption", []))
            if captions:
                return f"[Image: {', '.join(captions)}]"

        elif item_type == "table" and self.config.include_captions:
            captions = item.get("table_caption", [])
            if captions:
                return f"[Table: {', '.join(captions)}]"

        return ""

    def _extract_from_dict_source(
        self, dict_source: Dict, current_item_info: Dict
    ) -> str:
        """从基于字典的内容源中提取上下文

        Args:
            dict_source: 包含内容的字典
            current_item_info: 当前项目信息

        Returns:
            提取的上下文文本
        """
        # 处理不同的字典结构
        if "content" in dict_source:
            context = str(dict_source["content"])
        elif "text" in dict_source:
            context = str(dict_source["text"])
        else:
            # 尝试提取任何字符串值
            text_parts = []
            for value in dict_source.values():
                if isinstance(value, str):
                    text_parts.append(value)
            context = "\n".join(text_parts)

        return self._truncate_context(context)

    def _extract_from_text_source(
        self, text_source: str, current_item_info: Dict
    ) -> str:
        """从纯文本源中提取上下文

        Args:
            text_source: 纯文本内容
            current_item_info: 当前项目信息

        Returns:
            截断的文本上下文
        """
        return self._truncate_context(text_source)

    def _extract_from_text_chunks(
        self, text_chunks: List[str], current_item_info: Dict
    ) -> str:
        """从简单的文本块列表中提取上下文

        Args:
            text_chunks: 文本字符串列表
            current_item_info: 包含索引的当前项目信息

        Returns:
            来自周围块的上下文文本
        """
        current_index = current_item_info.get("index", 0)
        window_size = self.config.context_window

        start_idx = max(0, current_index - window_size)
        end_idx = min(len(text_chunks), current_index + window_size + 1)

        context_texts = []
        for i in range(start_idx, end_idx):
            if i != current_index:  # 排除当前块
                if i < len(text_chunks):
                    chunk_text = str(text_chunks[i]).strip()
                    if chunk_text:
                        context_texts.append(chunk_text)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _truncate_context(self, context: str) -> str:
        """将上下文截断到最大 token 限制

        Args:
            context: 要截断的上下文文本

        Returns:
            截断后的上下文文本
        """
        if not context:
            return ""

        # 如果可用，使用分词器进行精确的 token 计数
        if self.tokenizer:
            tokens = self.tokenizer.encode(context)
            if len(tokens) <= self.config.max_context_tokens:
                return context

            # 截断到最大 token 数并解码回文本
            truncated_tokens = tokens[: self.config.max_context_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)

            # 尝试在句子边界处结束
            last_period = truncated_text.rfind(".")
            last_newline = truncated_text.rfind("\n")

            if last_period > len(truncated_text) * 0.8:
                return truncated_text[: last_period + 1]
            elif last_newline > len(truncated_text) * 0.8:
                return truncated_text[:last_newline]
            else:
                return truncated_text + "..."
        else:
            # 如果没有分词器，回退到基于字符的截断
            if len(context) <= self.config.max_context_tokens:
                return context

            # 简单截断 - 没有分词器时的回退方案
            truncated = context[: self.config.max_context_tokens]

            # 尝试在句子边界处结束
            last_period = truncated.rfind(".")
            last_newline = truncated.rfind("\n")

            if last_period > len(truncated) * 0.8:
                return truncated[: last_period + 1]
            elif last_newline > len(truncated) * 0.8:
                return truncated[:last_newline]
            else:
                return truncated + "..."


class BaseModalProcessor:
    """模态处理器的基类"""

    def __init__(
        self,
        lightrag: LightRAG,
        modal_caption_func,
        context_extractor: ContextExtractor = None,
    ):
        """初始化基础处理器

        Args:
            lightrag: LightRAG 实例
            modal_caption_func: 用于生成描述的函数
            context_extractor: 上下文提取器实例
        """
        self.lightrag = lightrag
        self.modal_caption_func = modal_caption_func

        # 使用 LightRAG 的存储实例
        self.text_chunks_db = lightrag.text_chunks
        self.chunks_vdb = lightrag.chunks_vdb
        self.entities_vdb = lightrag.entities_vdb
        self.relationships_vdb = lightrag.relationships_vdb
        self.knowledge_graph_inst = lightrag.chunk_entity_relation_graph

        # 使用 LightRAG 的配置和函数
        self.embedding_func = lightrag.embedding_func
        self.llm_model_func = lightrag.llm_model_func
        self.global_config = asdict(lightrag)
        self.hashing_kv = lightrag.llm_response_cache
        self.tokenizer = lightrag.tokenizer

        # 如果未提供，则使用分词器初始化上下文提取器
        if context_extractor is None:
            self.context_extractor = ContextExtractor(tokenizer=self.tokenizer)
        else:
            self.context_extractor = context_extractor
            # 如果 context_extractor 没有分词器，则更新它
            if self.context_extractor.tokenizer is None:
                self.context_extractor.tokenizer = self.tokenizer

        # 用于上下文提取的内容源
        self.content_source = None
        self.content_format = "auto"

    def set_content_source(self, content_source: Any, content_format: str = "auto"):
        """设置用于上下文提取的内容源

        Args:
            content_source: 用于上下文提取的源内容
            content_format: 内容源的格式（"minerU"、"text_chunks"、"auto"）
        """
        self.content_source = content_source
        self.content_format = content_format
        logger.info(f"Content source set with format: {content_format}")

    def _get_context_for_item(self, item_info: Dict[str, Any]) -> str:
        """获取当前处理项目的上下文

        Args:
            item_info: 当前项目的信息（page_idx、index 等）

        Returns:
            项目的上下文文本
        """
        if not self.content_source:
            return ""

        try:
            context = self.context_extractor.extract_context(
                self.content_source, item_info, self.content_format
            )
            if context:
                logger.debug(
                    f"Extracted context of length {len(context)} for item: {item_info}"
                )
            return context
        except Exception as e:
            logger.error(f"Error getting context for item {item_info}: {e}")
            return ""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        仅生成文本描述和实体信息，不进行实体关系提取。
        用于批处理阶段 1。

        Args:
            modal_content: 要处理的模态内容
            content_type: 模态内容的类型
            item_info: 用于上下文提取的项目信息
            entity_name: 可选的预定义实体名称

        Returns:
            (描述, 实体信息) 的元组
        """
        # 子类必须实现此方法
        raise NotImplementedError("Subclasses must implement this method")

    async def _create_entity_and_chunk(
        self,
        modal_chunk: str,
        entity_info: Dict[str, Any],
        file_path: str,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """创建实体和文本块"""
        # 创建块
        chunk_id = compute_mdhash_id(str(modal_chunk), prefix="chunk-")
        tokens = len(self.tokenizer.encode(modal_chunk))

        # 使用提供的 doc_id 或从 chunk_id 生成一个以保持向后兼容性
        actual_doc_id = doc_id if doc_id else chunk_id

        chunk_data = {
            "tokens": tokens,
            "content": modal_chunk,
            "chunk_order_index": chunk_order_index,
            "full_doc_id": actual_doc_id,  # 使用正确的文档 ID
            "file_path": file_path,
        }

        # 存储块
        await self.text_chunks_db.upsert({chunk_id: chunk_data})

        # 在向量数据库中存储块以供检索
        chunk_vdb_data = {
            chunk_id: {
                "content": modal_chunk,
                "full_doc_id": actual_doc_id,
                "tokens": tokens,
                "chunk_order_index": chunk_order_index,
                "file_path": file_path,
            }
        }
        await self.chunks_vdb.upsert(chunk_vdb_data)

        # 创建实体节点
        node_data = {
            "entity_id": entity_info["entity_name"],
            "entity_type": entity_info["entity_type"],
            "description": entity_info["summary"],
            "source_id": chunk_id,
            "file_path": file_path,
            "created_at": int(time.time()),
        }

        await self.knowledge_graph_inst.upsert_node(
            entity_info["entity_name"], node_data
        )

        # 将实体插入向量数据库
        entity_vdb_data = {
            compute_mdhash_id(entity_info["entity_name"], prefix="ent-"): {
                "entity_name": entity_info["entity_name"],
                "entity_type": entity_info["entity_type"],
                "content": f"{entity_info['entity_name']}\n{entity_info['summary']}",
                "source_id": chunk_id,
                "file_path": file_path,
            }
        }
        await self.entities_vdb.upsert(entity_vdb_data)

        # 处理实体和关系提取
        chunk_results = await self._process_chunk_for_extraction(
            chunk_id, entity_info["entity_name"], batch_mode
        )

        return (
            entity_info["summary"],
            {
                "entity_name": entity_info["entity_name"],
                "entity_type": entity_info["entity_type"],
                "description": entity_info["summary"],
                "chunk_id": chunk_id,
            },
            chunk_results,
        )

    def _robust_json_parse(self, response: str) -> dict:
        """使用多种回退策略的鲁棒 JSON 解析"""

        # 策略 1：首先尝试直接解析
        for json_candidate in self._extract_all_json_candidates(response):
            result = self._try_parse_json(json_candidate)
            if result:
                return result

        # 策略 2：尝试基本清理
        for json_candidate in self._extract_all_json_candidates(response):
            cleaned = self._basic_json_cleanup(json_candidate)
            result = self._try_parse_json(cleaned)
            if result:
                return result

        # 策略 3：尝试渐进式引号修复
        for json_candidate in self._extract_all_json_candidates(response):
            fixed = self._progressive_quote_fix(json_candidate)
            result = self._try_parse_json(fixed)
            if result:
                return result

        # 策略 4：回退到正则表达式字段提取
        return self._extract_fields_with_regex(response)

    def _extract_all_json_candidates(self, response: str) -> list:
        """从响应中提取所有可能的 JSON 候选项"""
        candidates = []

        # 方法 1：代码块中的 JSON
        import re

        json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        candidates.extend(json_blocks)

        # 方法 2：平衡的大括号
        brace_count = 0
        start_pos = -1

        for i, char in enumerate(response):
            if char == "{":
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    candidates.append(response[start_pos : i + 1])

        # 方法 3：简单的正则表达式回退
        simple_match = re.search(r"\{.*\}", response, re.DOTALL)
        if simple_match:
            candidates.append(simple_match.group(0))

        return candidates

    def _try_parse_json(self, json_str: str) -> dict:
        """尝试解析 JSON 字符串，失败则返回 None"""
        if not json_str or not json_str.strip():
            return None

        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

    def _basic_json_cleanup(self, json_str: str) -> str:
        """对常见 JSON 问题进行基本清理"""
        # 移除多余的空白
        json_str = json_str.strip()

        # 修复常见的引号问题
        json_str = json_str.replace('"', '"').replace('"', '"')  # 智能引号
        json_str = json_str.replace(""", "'").replace(""", "'")  # 智能撇号

        # 修复尾随逗号（简单情况）
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        return json_str

    def _progressive_quote_fix(self, json_str: str) -> str:
        """渐进式修复引号和转义问题"""
        # 仅转义引号前未转义的反斜杠
        json_str = re.sub(r'(?<!\\)\\(?=")', r"\\\\", json_str)

        # 修复字符串值中未转义的反斜杠（更保守）
        def fix_string_content(match):
            content = match.group(1)
            # 仅转义明显有问题的模式
            content = re.sub(r"\\(?=[a-zA-Z])", r"\\\\", content)  # \alpha -> \\alpha
            return f'"{content}"'

        json_str = re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_string_content, json_str)
        return json_str

    def _extract_fields_with_regex(self, response: str) -> dict:
        """使用正则表达式作为最后手段提取所需字段"""
        logger.warning("Using regex fallback for JSON parsing")

        # 提取 detailed_description
        desc_match = re.search(
            r'"detailed_description":\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL
        )
        description = desc_match.group(1) if desc_match else ""

        # 提取 entity_name
        name_match = re.search(r'"entity_name":\s*"([^"]*(?:\\.[^"]*)*)"', response)
        entity_name = name_match.group(1) if name_match else "unknown_entity"

        # 提取 entity_type
        type_match = re.search(r'"entity_type":\s*"([^"]*(?:\\.[^"]*)*)"', response)
        entity_type = type_match.group(1) if type_match else "unknown"

        # 提取 summary
        summary_match = re.search(
            r'"summary":\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL
        )
        summary = summary_match.group(1) if summary_match else description[:100]

        return {
            "detailed_description": description,
            "entity_info": {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "summary": summary,
            },
        }

    def _extract_json_from_response(self, response: str) -> str:
        """遗留方法 - 现在由 _extract_all_json_candidates 处理"""
        candidates = self._extract_all_json_candidates(response)
        return candidates[0] if candidates else None

    def _fix_json_escapes(self, json_str: str) -> str:
        """遗留方法 - 现在由渐进式策略处理"""
        return self._progressive_quote_fix(json_str)

    async def _process_chunk_for_extraction(
        self, chunk_id: str, modal_entity_name: str, batch_mode: bool = False
    ):
        """处理块以进行实体和关系提取"""
        chunk_data = await self.text_chunks_db.get_by_id(chunk_id)
        if not chunk_data:
            logger.error(f"Chunk {chunk_id} not found")
            return

        # 为向量数据库创建文本块
        chunk_vdb_data = {
            chunk_id: {
                "content": chunk_data["content"],
                "full_doc_id": chunk_id,
                "tokens": chunk_data["tokens"],
                "chunk_order_index": chunk_data["chunk_order_index"],
                "file_path": chunk_data["file_path"],
            }
        }

        await self.chunks_vdb.upsert(chunk_vdb_data)

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # 准备用于提取的块
        chunks = {chunk_id: chunk_data}

        # 提取实体和关系
        chunk_results = await extract_entities(
            chunks=chunks,
            global_config=self.global_config,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.hashing_kv,
        )

        # 为所有提取的实体添加 "belongs_to" 关系
        processed_chunk_results = []
        for maybe_nodes, maybe_edges in chunk_results:
            for entity_name in maybe_nodes.keys():
                if entity_name != modal_entity_name:  # 跳过自我关系
                    # 创建 belongs_to 关系
                    relation_data = {
                        "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                        "keywords": "belongs_to,part_of,contained_in",
                        "source_id": chunk_id,
                        "weight": 10.0,
                        "file_path": chunk_data.get("file_path", "manual_creation"),
                    }
                    await self.knowledge_graph_inst.upsert_edge(
                        entity_name, modal_entity_name, relation_data
                    )

                    relation_id = compute_mdhash_id(
                        entity_name + modal_entity_name, prefix="rel-"
                    )
                    relation_vdb_data = {
                        relation_id: {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "keywords": relation_data["keywords"],
                            "content": f"{relation_data['keywords']}\t{entity_name}\n{modal_entity_name}\n{relation_data['description']}",
                            "source_id": chunk_id,
                            "file_path": chunk_data.get("file_path", "manual_creation"),
                        }
                    }
                    await self.relationships_vdb.upsert(relation_vdb_data)

                    # 添加到 maybe_edges
                    maybe_edges[(entity_name, modal_entity_name)] = [relation_data]

            processed_chunk_results.append((maybe_nodes, maybe_edges))

        if not batch_mode:
            # 使用正确的 file_path 参数进行合并
            file_path = chunk_data.get("file_path", "manual_creation")
            await merge_nodes_and_edges(
                chunk_results=chunk_results,
                knowledge_graph_inst=self.knowledge_graph_inst,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=self.global_config,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.hashing_kv,
                current_file_number=1,
                total_files=1,
                file_path=file_path,  # 传递正确的 file_path
            )

            # 确保所有存储更新都已完成
            await self.lightrag._insert_done()

        return processed_chunk_results


class ImageModalProcessor(BaseModalProcessor):
    """图像内容的专用处理器"""

    def __init__(
        self,
        lightrag: LightRAG,
        modal_caption_func,
        context_extractor: ContextExtractor = None,
    ):
        """初始化图像处理器

        Args:
            lightrag: LightRAG 实例
            modal_caption_func: 用于生成描述的函数（支持图像理解）
            context_extractor: 上下文提取器实例
        """
        super().__init__(lightrag, modal_caption_func, context_extractor)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图像编码为 base64"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        仅生成图像描述和实体信息，不进行实体关系提取。
        用于批处理阶段 1。

        Args:
            modal_content: 要处理的图像内容
            content_type: 模态内容的类型（"image"）
            item_info: 用于上下文提取的项目信息
            entity_name: 可选的预定义实体名称

        Returns:
            (enhanced_caption, entity_info) 的元组
        """
        try:
            # 解析图像内容（重用现有逻辑）
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content

            image_path = content_data.get("img_path")
            captions = content_data.get(
                "image_caption", content_data.get("img_caption", [])
            )
            footnotes = content_data.get(
                "image_footnote", content_data.get("img_footnote", [])
            )

            # 验证图像路径
            if not image_path:
                raise ValueError(
                    f"No image path provided in modal_content: {modal_content}"
                )

            # 转换为 Path 对象并检查是否存在
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # 提取当前项目的上下文
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # 构建带上下文的详细视觉分析提示词
            if context:
                vision_prompt = PROMPTS.get(
                    "vision_prompt_with_context", PROMPTS["vision_prompt"]
                ).format(
                    context=context,
                    entity_name=entity_name
                    if entity_name
                    else "unique descriptive name for this image",
                    image_path=image_path,
                    captions=captions if captions else "None",
                    footnotes=footnotes if footnotes else "None",
                )
            else:
                vision_prompt = PROMPTS["vision_prompt"].format(
                    entity_name=entity_name
                    if entity_name
                    else "unique descriptive name for this image",
                    image_path=image_path,
                    captions=captions if captions else "None",
                    footnotes=footnotes if footnotes else "None",
                )

            # 将图像编码为 base64
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                raise RuntimeError(f"Failed to encode image to base64: {image_path}")

            # 使用编码的图像调用视觉模型
            response = await self.modal_caption_func(
                vision_prompt,
                image_data=image_base64,
                system_prompt=PROMPTS["IMAGE_ANALYSIS_SYSTEM"],
            )

            # 解析响应（重用现有逻辑）
            enhanced_caption, entity_info = self._parse_response(response, entity_name)

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"image_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "image",
                "summary": f"Image content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """处理带有上下文支持的图像内容"""
        try:
            # 生成描述和实体信息
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # 构建完整的图像内容
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content

            image_path = content_data.get("img_path", "")
            captions = content_data.get(
                "image_caption", content_data.get("img_caption", [])
            )
            footnotes = content_data.get(
                "image_footnote", content_data.get("img_footnote", [])
            )

            modal_chunk = PROMPTS["image_chunk"].format(
                image_path=image_path,
                captions=", ".join(captions) if captions else "None",
                footnotes=", ".join(footnotes) if footnotes else "None",
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing image content: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"image_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "image",
                "summary": f"Image content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    def _parse_response(
        self, response: str, entity_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """解析模型响应"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing image analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"image_{compute_mdhash_id(response)}",
                "entity_type": "image",
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity


class TableModalProcessor(BaseModalProcessor):
    """表格内容的专用处理器"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        仅生成表格描述和实体信息，不进行实体关系提取。
        用于批处理阶段 1。

        Args:
            modal_content: 要处理的表格内容
            content_type: 模态内容的类型（"table"）
            item_info: 用于上下文提取的项目信息
            entity_name: 可选的预定义实体名称

        Returns:
            (enhanced_caption, entity_info) 的元组
        """
        try:
            # 解析表格内容（重用现有逻辑）
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"table_body": modal_content}
            else:
                content_data = modal_content

            table_img_path = content_data.get("img_path")
            table_caption = content_data.get("table_caption", [])
            table_body = content_data.get("table_body", "")
            table_footnote = content_data.get("table_footnote", [])

            # 提取当前项目的上下文
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # 构建带上下文的表格分析提示词
            if context:
                table_prompt = PROMPTS.get(
                    "table_prompt_with_context", PROMPTS["table_prompt"]
                ).format(
                    context=context,
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this table",
                    table_img_path=table_img_path,
                    table_caption=table_caption if table_caption else "None",
                    table_body=table_body,
                    table_footnote=table_footnote if table_footnote else "None",
                )
            else:
                table_prompt = PROMPTS["table_prompt"].format(
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this table",
                    table_img_path=table_img_path,
                    table_caption=table_caption if table_caption else "None",
                    table_body=table_body,
                    table_footnote=table_footnote if table_footnote else "None",
                )

            # 调用 LLM 进行表格分析
            response = await self.modal_caption_func(
                table_prompt,
                system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"],
            )

            # 解析响应（重用现有逻辑）
            enhanced_caption, entity_info = self._parse_table_response(
                response, entity_name
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating table description: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"table_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "table",
                "summary": f"Table content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """处理带有上下文支持的表格内容"""
        try:
            # 生成描述和实体信息
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # 解析表格内容以构建完整的块
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"table_body": modal_content}
            else:
                content_data = modal_content

            table_img_path = content_data.get("img_path")
            table_caption = content_data.get("table_caption", [])
            table_body = content_data.get("table_body", "")
            table_footnote = content_data.get("table_footnote", [])

            # 构建完整的表格内容
            modal_chunk = PROMPTS["table_chunk"].format(
                table_img_path=table_img_path,
                table_caption=", ".join(table_caption) if table_caption else "None",
                table_body=table_body,
                table_footnote=", ".join(table_footnote) if table_footnote else "None",
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing table content: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"table_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "table",
                "summary": f"Table content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    def _parse_table_response(
        self, response: str, entity_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """解析表格分析响应"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing table analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"table_{compute_mdhash_id(response)}",
                "entity_type": "table",
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity


class EquationModalProcessor(BaseModalProcessor):
    """公式内容的专用处理器"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        仅生成公式描述和实体信息，不进行实体关系提取。
        用于批处理阶段 1。

        Args:
            modal_content: 要处理的公式内容
            content_type: 模态内容的类型（"equation"）
            item_info: 用于上下文提取的项目信息
            entity_name: 可选的预定义实体名称

        Returns:
            (enhanced_caption, entity_info) 的元组
        """
        try:
            # 解析公式内容（重用现有逻辑）
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"equation": modal_content}
            else:
                content_data = modal_content

            equation_text = content_data.get("text")
            equation_format = content_data.get("text_format", "")

            # 提取当前项目的上下文
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # 构建带上下文的公式分析提示词
            if context:
                equation_prompt = PROMPTS.get(
                    "equation_prompt_with_context", PROMPTS["equation_prompt"]
                ).format(
                    context=context,
                    equation_text=equation_text,
                    equation_format=equation_format,
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this equation",
                )
            else:
                equation_prompt = PROMPTS["equation_prompt"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    entity_name=entity_name
                    if entity_name
                    else "descriptive name for this equation",
                )

            # 调用 LLM 进行公式分析
            response = await self.modal_caption_func(
                equation_prompt,
                system_prompt=PROMPTS["EQUATION_ANALYSIS_SYSTEM"],
            )

            # 解析响应（重用现有逻辑）
            enhanced_caption, entity_info = self._parse_equation_response(
                response, entity_name
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating equation description: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"equation_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "equation",
                "summary": f"Equation content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """处理带有上下文支持的公式内容"""
        try:
            # 生成描述和实体信息
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # 解析公式内容以构建完整的块
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"equation": modal_content}
            else:
                content_data = modal_content

            equation_text = content_data.get("text")
            equation_format = content_data.get("text_format", "")

            # 构建完整的公式内容
            modal_chunk = PROMPTS["equation_chunk"].format(
                equation_text=equation_text,
                equation_format=equation_format,
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing equation content: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"equation_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "equation",
                "summary": f"Equation content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    def _parse_equation_response(
        self, response: str, entity_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """使用鲁棒 JSON 处理解析公式分析响应"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing equation analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"equation_{compute_mdhash_id(response)}",
                "entity_type": "equation",
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity


class GenericModalProcessor(BaseModalProcessor):
    """其他类型模态内容的通用处理器"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        仅生成通用模态描述和实体信息，不进行实体关系提取。
        用于批处理阶段 1。

        Args:
            modal_content: 要处理的通用模态内容
            content_type: 模态内容的类型
            item_info: 用于上下文提取的项目信息
            entity_name: 可选的预定义实体名称

        Returns:
            (enhanced_caption, entity_info) 的元组
        """
        try:
            # 提取当前项目的上下文
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # 构建带上下文的通用分析提示词
            if context:
                generic_prompt = PROMPTS.get(
                    "generic_prompt_with_context", PROMPTS["generic_prompt"]
                ).format(
                    context=context,
                    content_type=content_type,
                    entity_name=entity_name
                    if entity_name
                    else f"descriptive name for this {content_type}",
                    content=str(modal_content),
                )
            else:
                generic_prompt = PROMPTS["generic_prompt"].format(
                    content_type=content_type,
                    entity_name=entity_name
                    if entity_name
                    else f"descriptive name for this {content_type}",
                    content=str(modal_content),
                )

            # 调用 LLM 进行通用分析
            response = await self.modal_caption_func(
                generic_prompt,
                system_prompt=PROMPTS["GENERIC_ANALYSIS_SYSTEM"].format(
                    content_type=content_type
                ),
            )

            # 解析响应（重用现有逻辑）
            enhanced_caption, entity_info = self._parse_generic_response(
                response, entity_name, content_type
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating {content_type} description: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(str(modal_content))}",
                "entity_type": content_type,
                "summary": f"{content_type} content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """处理带有上下文支持的通用模态内容"""
        try:
            # 生成描述和实体信息
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # 构建完整的内容
            modal_chunk = PROMPTS["generic_chunk"].format(
                content_type=content_type.title(),
                content=str(modal_content),
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing {content_type} content: {e}")
            # 回退处理
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(str(modal_content))}",
                "entity_type": content_type,
                "summary": f"{content_type} content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    def _parse_generic_response(
        self, response: str, entity_name: str = None, content_type: str = "content"
    ) -> Tuple[str, Dict[str, Any]]:
        """解析通用分析响应"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing {content_type} analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(response)}",
                "entity_type": content_type,
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity
