"""
RAGAnything 的文档处理功能

包含用于解析文档和处理多模态内容的方法
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from raganything.base import DocStatus
from raganything.parser import MineruParser, DoclingParser, MineruExecutionError
from raganything.utils import (
    separate_content,
    insert_text_content,
    insert_text_content_with_multimodal_content,
    get_processor_for_type,
)
import asyncio
from lightrag.utils import compute_mdhash_id


class ProcessorMixin:
    """包含 RAGAnything 文档处理功能的 ProcessorMixin 类"""

    def _generate_cache_key(
        self, file_path: Path, parse_method: str = None, **kwargs
    ) -> str:
        """
        基于文件路径和解析配置生成缓存键

        Args:
            file_path: 文件路径
            parse_method: 使用的解析方法
            **kwargs: 额外的解析器参数

        Returns:
            str: 文件和配置的缓存键
        """

        # 获取文件修改时间
        mtime = file_path.stat().st_mtime

        # 创建缓存键的配置字典
        config_dict = {
            "file_path": str(file_path.absolute()),
            "mtime": mtime,
            "parser": self.config.parser,
            "parse_method": parse_method or self.config.parse_method,
        }

        # 将相关的 kwargs 添加到配置中
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "lang",
                "device",
                "start_page",
                "end_page",
                "formula",
                "table",
                "backend",
                "source",
            ]
        }
        config_dict.update(relevant_kwargs)

        # 从配置生成哈希值
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """
        基于文档内容生成 doc_id

        Args:
            content_list: 解析后的内容列表

        Returns:
            str: 带有 doc- 前缀的基于内容的文档 ID
        """
        from lightrag.utils import compute_mdhash_id

        # 提取用于 ID 生成的关键内容
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # 对于文本内容，使用文本
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # 对于其他内容类型，使用关键标识符
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # 对于其他类型，使用字符串表示
                    content_hash_data.append(str(item))

        # 创建内容签名
        content_signature = "\n".join(content_hash_data)

        # 从内容生成 doc_id
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def _get_cached_result(
        self, cache_key: str, file_path: Path, parse_method: str = None, **kwargs
    ) -> tuple[List[Dict[str, Any]], str] | None:
        """
        如果缓存可用且有效，获取缓存的解析结果

        Args:
            cache_key: 要查找的缓存键
            file_path: 用于 mtime 检查的文件路径
            parse_method: 使用的解析方法
            **kwargs: 额外的解析器参数

        Returns:
            tuple[List[Dict[str, Any]], str] | None: (content_list, doc_id) 或 None（如果未找到/无效）
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return None

        try:
            cached_data = await self.parse_cache.get_by_id(cache_key)
            if not cached_data:
                return None

            # 检查文件修改时间
            current_mtime = file_path.stat().st_mtime
            cached_mtime = cached_data.get("mtime", 0)

            if current_mtime != cached_mtime:
                self.logger.debug(f"Cache invalid - file modified: {cache_key}")
                return None

            # 检查解析配置
            cached_config = cached_data.get("parse_config", {})
            current_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # 将相关的 kwargs 添加到当前配置中
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            current_config.update(relevant_kwargs)

            if cached_config != current_config:
                self.logger.debug(f"Cache invalid - config changed: {cache_key}")
                return None

            content_list = cached_data.get("content_list", [])
            doc_id = cached_data.get("doc_id")

            if content_list and doc_id:
                self.logger.debug(
                    f"Found valid cached parsing result for key: {cache_key}"
                )
                return content_list, doc_id
            else:
                self.logger.debug(
                    f"Cache incomplete - missing content or doc_id: {cache_key}"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error accessing parse cache: {e}")

        return None

    async def _store_cached_result(
        self,
        cache_key: str,
        content_list: List[Dict[str, Any]],
        doc_id: str,
        file_path: Path,
        parse_method: str = None,
        **kwargs,
    ) -> None:
        """
        将解析结果存储到缓存中

        Args:
            cache_key: 存储时使用的缓存键
            content_list: 要缓存的内容列表
            doc_id: 基于内容的文档 ID
            file_path: 用于 mtime 存储的文件路径
            parse_method: 使用的解析方法
            **kwargs: 额外的解析器参数
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return

        try:
            # 获取文件修改时间
            file_mtime = file_path.stat().st_mtime

            # 创建解析配置
            parse_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # 将相关的 kwargs 添加到配置中
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            parse_config.update(relevant_kwargs)

            cache_data = {
                cache_key: {
                    "content_list": content_list,
                    "doc_id": doc_id,
                    "mtime": file_mtime,
                    "parse_config": parse_config,
                    "cached_at": time.time(),
                    "cache_version": "1.0",
                }
            }
            await self.parse_cache.upsert(cache_data)
            # 确保数据持久化到磁盘
            await self.parse_cache.index_done_callback()
            self.logger.info(f"Stored parsing result in cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error storing to parse cache: {e}")

    async def parse_document(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        **kwargs,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        带缓存支持的文档解析

        Args:
            file_path: 要解析的文件路径
            output_dir: 输出目录（默认为 config.parser_output_dir）
            parse_method: 解析方法（默认为 config.parse_method）
            display_stats: 是否显示内容统计（默认为 config.display_content_stats）
            **kwargs: 解析器的额外参数（例如：lang, device, start_page, end_page, formula, table, backend, source）

        Returns:
            tuple[List[Dict[str, Any]], str]: (content_list, doc_id)
        """
        # 如果未提供，使用配置默认值
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting document parsing: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 基于文件和配置生成缓存键
        cache_key = self._generate_cache_key(file_path, parse_method, **kwargs)

        # 首先检查缓存
        cached_result = await self._get_cached_result(
            cache_key, file_path, parse_method, **kwargs
        )
        if cached_result is not None:
            content_list, doc_id = cached_result
            self.logger.info(f"Using cached parsing result for: {file_path}")
            if display_stats:
                self.logger.info(
                    f"* Total blocks in cached content_list: {len(content_list)}"
                )
            return content_list, doc_id

        # 根据文件扩展名选择适当的解析方法
        ext = file_path.suffix.lower()

        try:
            doc_parser = (
                DoclingParser() if self.config.parser == "docling" else MineruParser()
            )

            # 记录解析器和方法信息
            self.logger.info(
                f"Using {self.config.parser} parser with method: {parse_method}"
            )

            if ext in [".pdf"]:
                self.logger.info("Detected PDF file, using parser for PDF...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    output_dir=output_dir,
                    method=parse_method,
                    **kwargs,
                )
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            ]:
                self.logger.info("Detected image file, using parser for images...")
                # 使用所选解析器的图像解析功能
                if hasattr(doc_parser, "parse_image"):
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                else:
                    # 如果当前解析器不支持图像解析，回退到 MinerU
                    self.logger.warning(
                        f"{self.config.parser} parser doesn't support image parsing, falling back to MinerU"
                    )
                    content_list = MineruParser().parse_image(
                        image_path=file_path, output_dir=output_dir, **kwargs
                    )
            elif ext in [
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".html",
                ".htm",
                ".xhtml",
            ]:
                self.logger.info(
                    "Detected Office or HTML document, using parser for Office/HTML..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=file_path,
                    output_dir=output_dir,
                    **kwargs,
                )
            else:
                # 对于其他或未知格式，使用通用解析器
                self.logger.info(
                    f"Using generic parser for {ext} file (method={parse_method})..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_document,
                    file_path=file_path,
                    method=parse_method,
                    output_dir=output_dir,
                    **kwargs,
                )

        except MineruExecutionError as e:
            self.logger.error(f"Mineru command failed: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error during parsing with {self.config.parser} parser: {str(e)}"
            )
            raise e

        msg = f"Parsing {file_path} complete! Extracted {len(content_list)} content blocks"
        self.logger.info(msg)

        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content was extracted")

        # 基于内容生成 doc_id
        doc_id = self._generate_content_based_doc_id(content_list)

        # 将结果存储到缓存中
        await self._store_cached_result(
            cache_key, content_list, doc_id, file_path, parse_method, **kwargs
        )

        # 如果请求，显示内容统计
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # 按类型计数元素
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        return content_list, doc_id

    async def _process_multimodal_content(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        pipeline_status: Optional[Any] = None,
        pipeline_status_lock: Optional[Any] = None,
    ):
        """
        处理多模态内容（使用专用处理器）

        Args:
            multimodal_items: 多模态项列表
            file_path: 文件路径（用于引用）
            doc_id: 用于正确关联块的文档 ID
            pipeline_status: 流水线状态对象
            pipeline_status_lock: 流水线状态锁
        """

        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # 检查多模态处理状态 - 处理 LightRAG 的早期 DocStatus.PROCESSED 标记
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if existing_doc_status:
                # 检查多模态内容是否已处理
                multimodal_processed = existing_doc_status.get(
                    "multimodal_processed", False
                )

                if multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} multimodal content is already processed"
                    )
                    return

                # 即使状态是 DocStatus.PROCESSED（文本处理完成），
                # 如果多模态内容尚未完成，我们仍然需要处理它
                doc_status = existing_doc_status.get("status", "")
                if doc_status == DocStatus.PROCESSED and not multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} text processing is complete, but multimodal content still needs processing"
                    )
                    # 继续多模态处理
                elif doc_status == DocStatus.PROCESSED and multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} is fully processed (text + multimodal)"
                    )
                    return

        except Exception as e:
            self.logger.debug(f"Error checking document status for {doc_id}: {e}")
            # 如果缓存检查失败，继续处理

        # 使用 ProcessorMixin 自己的批处理，可以处理多种内容类型
        log_message = "Starting multimodal content processing..."
        self.logger.info(log_message)
        if pipeline_status_lock and pipeline_status:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        try:
            # 确保 LightRAG 已初始化
            await self._ensure_lightrag_initialized()

            await self._process_multimodal_content_batch_type_aware(
                multimodal_items=multimodal_items, file_path=file_path, doc_id=doc_id
            )

            # 标记多模态内容已处理并更新最终状态
            await self._mark_multimodal_processing_complete(doc_id)

            log_message = "Multimodal content processing complete"
            self.logger.info(log_message)
            if pipeline_status_lock and pipeline_status:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

        except Exception as e:
            self.logger.error(f"Error in multimodal processing: {e}")
            # 如果批处理失败，回退到单个处理
            self.logger.warning("Falling back to individual multimodal processing")
            await self._process_multimodal_content_individual(
                multimodal_items, file_path, doc_id
            )

            # 即使在回退后也标记多模态内容已处理
            await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_individual(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        单独处理多模态内容（回退方法）

        Args:
            multimodal_items: 多模态项列表
            file_path: 文件路径（用于引用）
            doc_id: 用于正确关联块的文档 ID
        """
        file_name = os.path.basename(file_path)

        # 收集所有块结果用于批处理（类似于文本内容处理）
        all_chunk_results = []
        multimodal_chunk_ids = []

        # 获取当前文本块数量以为多模态块设置正确的顺序索引
        existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
        existing_chunks_count = (
            existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
        )

        for i, item in enumerate(multimodal_items):
            try:
                content_type = item.get("type", "unknown")
                self.logger.info(
                    f"Processing item {i+1}/{len(multimodal_items)}: {content_type} content"
                )

                # 选择适当的处理器
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # 准备项信息用于上下文提取
                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": i,
                        "type": content_type,
                    }

                    # 处理内容并获取块结果而不是立即合并
                    (
                        enhanced_caption,
                        entity_info,
                        chunk_results,
                    ) = await processor.process_multimodal_content(
                        modal_content=item,
                        content_type=content_type,
                        file_path=file_name,
                        item_info=item_info,  # 传递项信息用于上下文提取
                        batch_mode=True,
                        doc_id=doc_id,  # 传递 doc_id 用于正确关联
                        chunk_order_index=existing_chunks_count
                        + i,  # 正确的顺序索引
                    )

                    # 收集块结果用于批处理
                    all_chunk_results.extend(chunk_results)

                    # 从 entity_info 中提取块 ID（处理器创建的实际 chunk_id）
                    if entity_info and "chunk_id" in entity_info:
                        chunk_id = entity_info["chunk_id"]
                        multimodal_chunk_ids.append(chunk_id)

                    self.logger.info(
                        f"{content_type} processing complete: {entity_info.get('entity_name', 'Unknown')}"
                    )
                else:
                    self.logger.warning(
                        f"No suitable processor found for {content_type} type content"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                self.logger.debug("Exception details:", exc_info=True)
                continue

        # 更新 doc_status 以在标准 chunks_list 中包含多模态块
        if multimodal_chunk_ids:
            try:
                # 获取当前文档状态
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

                if current_doc_status:
                    existing_chunks_list = current_doc_status.get("chunks_list", [])
                    existing_chunks_count = current_doc_status.get("chunks_count", 0)

                    # 将多模态块添加到标准 chunks_list
                    updated_chunks_list = existing_chunks_list + multimodal_chunk_ids
                    updated_chunks_count = existing_chunks_count + len(
                        multimodal_chunk_ids
                    )

                    # 使用集成的块列表更新文档状态
                    await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,  # 保留现有字段
                                "chunks_list": updated_chunks_list,  # 集成的块列表
                                "chunks_count": updated_chunks_count,  # 更新的总计数
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )

                    # 确保 doc_status 更新持久化到磁盘
                    await self.lightrag.doc_status.index_done_callback()

                    self.logger.info(
                        f"Updated doc_status with {len(multimodal_chunk_ids)} multimodal chunks integrated into chunks_list"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Error updating doc_status with multimodal chunks: {e}"
                )

        # 批量合并所有多模态内容结果（类似于文本内容处理）
        if all_chunk_results:
            from lightrag.operate import merge_nodes_and_edges
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            # 从共享存储获取流水线状态和锁
            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            await merge_nodes_and_edges(
                chunk_results=all_chunk_results,
                knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
                entity_vdb=self.lightrag.entities_vdb,
                relationships_vdb=self.lightrag.relationships_vdb,
                global_config=self.lightrag.__dict__,
                full_entities_storage=self.lightrag.full_entities,
                full_relations_storage=self.lightrag.full_relations,
                doc_id=doc_id,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.lightrag.llm_response_cache,
                current_file_number=1,
                total_files=1,
                file_path=file_name,
            )

            await self.lightrag._insert_done()

        self.logger.info("Individual multimodal content processing complete")

        # 标记多模态内容已处理
        await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_batch_type_aware(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        类型感知的批处理，根据内容类型选择正确的处理器。
        这是正确处理不同模态类型的实现。

        Args:
            multimodal_items: 具有不同类型的多模态项列表
            file_path: 用于引用的文件路径
            doc_id: 用于正确关联的文档 ID
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # 获取现有块数量以进行正确的顺序索引
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            existing_chunks_count = (
                existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
            )
        except Exception:
            existing_chunks_count = 0

        # 使用 LightRAG 的并发控制
        semaphore = asyncio.Semaphore(getattr(self.lightrag, "max_parallel_insert", 2))

        # 进度跟踪变量
        total_items = len(multimodal_items)
        completed_count = 0
        progress_lock = asyncio.Lock()

        # 记录处理开始
        self.logger.info(f"Starting to process {total_items} multimodal content items")

        # 阶段 1: 使用每种类型的正确处理器并发生成描述
        async def process_single_item_with_correct_processor(
            item: Dict[str, Any], index: int, file_path: str
        ):
            """使用正确的处理器处理单个项"""
            nonlocal completed_count
            async with semaphore:
                try:
                    content_type = item.get("type", "unknown")

                    # 根据内容类型选择正确的处理器
                    processor = get_processor_for_type(
                        self.modal_processors, content_type
                    )

                    if not processor:
                        self.logger.warning(
                            f"No processor found for type: {content_type}"
                        )
                        return None

                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": index,
                        "type": content_type,
                    }

                    # 调用正确处理器的描述生成方法
                    (
                        description,
                        entity_info,
                    ) = await processor.generate_description_only(
                        modal_content=item,
                        content_type=content_type,
                        item_info=item_info,
                        entity_name=None,  # 让 LLM 自动生成
                    )

                    # 更新进度（非阻塞）
                    async with progress_lock:
                        completed_count += 1
                        if (
                            completed_count % max(1, total_items // 10) == 0
                            or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    return {
                        "index": index,
                        "content_type": content_type,
                        "description": description,
                        "entity_info": entity_info,
                        "original_item": item,
                        "item_info": item_info,
                        "chunk_order_index": existing_chunks_count + index,
                        "processor": processor,  # 保留所使用处理器的引用
                        "file_path": file_path,  # 将 file_path 添加到结果中
                    }

                except Exception as e:
                    # 即使在错误时也更新进度（非阻塞）
                    async with progress_lock:
                        completed_count += 1
                        if (
                            completed_count % max(1, total_items // 10) == 0
                            or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    self.logger.error(
                        f"Error generating description for {content_type} item {index}: {e}"
                    )
                    return None

        # 使用正确的处理器并发处理所有项
        tasks = [
            asyncio.create_task(
                process_single_item_with_correct_processor(item, i, file_path)
            )
            for i, item in enumerate(multimodal_items)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤成功的结果
        multimodal_data_list = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                continue
            if result is not None:
                multimodal_data_list.append(result)

        if not multimodal_data_list:
            self.logger.warning("No valid multimodal descriptions generated")
            return

        self.logger.info(
            f"Generated descriptions for {len(multimodal_data_list)}/{len(multimodal_items)} multimodal items using correct processors"
        )

        # 阶段 2: 转换为 LightRAG 块格式
        lightrag_chunks = self._convert_to_lightrag_chunks_type_aware(
            multimodal_data_list, file_path, doc_id
        )

        # 阶段 3: 将块存储到 LightRAG 存储
        await self._store_chunks_to_lightrag_storage_type_aware(lightrag_chunks)

        # 阶段 3.5: 将多模态主实体存储到 entities_vdb 和 full_entities
        await self._store_multimodal_main_entities(
            multimodal_data_list, lightrag_chunks, file_path, doc_id
        )

        # 跟踪块 ID 以更新 doc_status
        chunk_ids = list(lightrag_chunks.keys())

        # 阶段 4: 使用 LightRAG 的批量实体关系提取
        chunk_results = await self._batch_extract_entities_lightrag_style_type_aware(
            lightrag_chunks
        )

        # 阶段 5: 添加 belongs_to 关系（多模态特定）
        enhanced_chunk_results = await self._batch_add_belongs_to_relations_type_aware(
            chunk_results, multimodal_data_list
        )

        # 阶段 6: 使用 LightRAG 的批量合并
        await self._batch_merge_lightrag_style_type_aware(
            enhanced_chunk_results, file_path, doc_id
        )

        # 阶段 7: 使用集成的 chunks_list 更新 doc_status
        await self._update_doc_status_with_chunks_type_aware(doc_id, chunk_ids)

    def _convert_to_lightrag_chunks_type_aware(
        self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str
    ) -> Dict[str, Any]:
        """将多模态数据转换为 LightRAG 标准块格式"""

        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # 根据内容类型应用适当的块模板
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # 生成 chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # 计算令牌数
            tokens = len(self.lightrag.tokenizer.encode(formatted_chunk_content))

            # 构建 LightRAG 标准块格式
            chunks[chunk_id] = {
                "content": formatted_chunk_content,  # 现在使用模板化内容
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": os.path.basename(file_path),
                "llm_cache_list": [],  # LightRAG 将填充此字段
                # 多模态特定元数据
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": data["content_type"],
                "page_idx": data["item_info"].get("page_idx", 0),
            }

        self.logger.debug(
            f"Converted {len(chunks)} multimodal items to multimodal chunks format"
        )
        return chunks

    def _apply_chunk_template(
        self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """
        根据内容类型应用适当的块模板

        Args:
            content_type: 内容类型（image, table, equation, generic）
            original_item: 原始多模态项数据
            description: 处理器生成的增强描述

        Returns:
            使用适当模板格式化的块内容
        """
        from raganything.prompt import PROMPTS

        try:
            if content_type == "image":
                image_path = original_item.get("img_path", "")
                captions = original_item.get(
                    "image_caption", original_item.get("img_caption", [])
                )
                footnotes = original_item.get(
                    "image_footnote", original_item.get("img_footnote", [])
                )

                return PROMPTS["image_chunk"].format(
                    image_path=image_path,
                    captions=", ".join(captions) if captions else "None",
                    footnotes=", ".join(footnotes) if footnotes else "None",
                    enhanced_caption=description,
                )

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_caption = original_item.get("table_caption", [])
                table_body = original_item.get("table_body", "")
                table_footnote = original_item.get("table_footnote", [])

                return PROMPTS["table_chunk"].format(
                    table_img_path=table_img_path,
                    table_caption=", ".join(table_caption) if table_caption else "None",
                    table_body=table_body,
                    table_footnote=", ".join(table_footnote)
                    if table_footnote
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "equation":
                equation_text = original_item.get("text", "")
                equation_format = original_item.get("text_format", "")

                return PROMPTS["equation_chunk"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    enhanced_caption=description,
                )

            else:  # 通用或未知类型
                content = str(original_item.get("content", original_item))

                return PROMPTS["generic_chunk"].format(
                    content_type=content_type.title(),
                    content=content,
                    enhanced_caption=description,
                )

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            # 如果模板失败，回退到仅使用描述
            return description

    async def _store_chunks_to_lightrag_storage_type_aware(
        self, chunks: Dict[str, Any]
    ):
        """将块存储到存储中"""
        try:
            # 存储在 text_chunks 存储中（extract_entities 所需）
            await self.lightrag.text_chunks.upsert(chunks)

            # 存储在块向量数据库中以供检索
            await self.lightrag.chunks_vdb.upsert(chunks)

            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")

        except Exception as e:
            self.logger.error(f"Error storing chunks to storage: {e}")
            raise

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        lightrag_chunks: Dict[str, Any],
        file_path: str,
        doc_id: str = None,
    ):
        """
        将多模态主实体存储到 entities_vdb 和 full_entities。
        这确保诸如"TableName (table)"之类的实体被正确索引。

        Args:
            multimodal_data_list: 带有实体信息的已处理多模态数据列表
            lightrag_chunks: LightRAG 格式的块（已使用模板格式化）
            file_path: 实体的文件路径
            doc_id: full_entities 存储的文档 ID
        """
        if not multimodal_data_list:
            return

        # 为所有多模态主实体创建 entities_vdb 条目
        entities_to_store = {}

        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # 应用相同的块模板以获取格式化内容
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # 使用格式化内容生成 chunk_id（与 _convert_to_lightrag_chunks 中相同）
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # 使用 LightRAG 的标准格式生成 entity_id
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # 以 LightRAG 格式创建实体数据
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_info.get("entity_type", content_type),
                "content": entity_info.get("summary", description),
                "source_id": chunk_id,
                "file_path": os.path.basename(file_path),
            }

            entities_to_store[entity_id] = entity_data

        if entities_to_store:
            try:
                # 在知识图谱中存储实体
                for entity_id, entity_data in entities_to_store.items():
                    entity_name = entity_data["entity_name"]

                    # 为知识图谱创建节点数据
                    node_data = {
                        "entity_id": entity_name,
                        "entity_type": entity_data["entity_type"],
                        "description": entity_data["content"],
                        "source_id": entity_data["source_id"],
                        "file_path": entity_data["file_path"],
                        "created_at": int(time.time()),
                    }

                    # 存储在知识图谱中
                    await self.lightrag.chunk_entity_relation_graph.upsert_node(
                        entity_name, node_data
                    )

                # 存储在 entities_vdb 中
                await self.lightrag.entities_vdb.upsert(entities_to_store)
                await self.lightrag.entities_vdb.index_done_callback()

                # 新增: 将多模态主实体存储在 full_entities 存储中
                if doc_id and self.lightrag.full_entities:
                    await self._store_multimodal_entities_to_full_entities(
                        entities_to_store, doc_id
                    )

                self.logger.debug(
                    f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
                )

            except Exception as e:
                self.logger.error(f"Error storing multimodal main entities: {e}")
                raise

    async def _store_multimodal_entities_to_full_entities(
        self, entities_to_store: Dict[str, Any], doc_id: str
    ):
        """
        将多模态主实体存储到 full_entities 存储中。

        Args:
            entities_to_store: 要存储的实体字典
            doc_id: 用于分组实体的文档 ID
        """
        try:
            # 获取此文档的当前 full_entities 数据
            current_doc_entities = await self.lightrag.full_entities.get_by_id(doc_id)

            if current_doc_entities is None:
                # 创建新文档条目
                entity_names = list(
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                )
                doc_entities_data = {
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            else:
                # 更新现有文档条目
                existing_entity_names = set(
                    current_doc_entities.get("entity_names", [])
                )
                new_entity_names = [
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                ]

                # 将新的多模态实体添加到列表中（避免重复）
                for entity_name in new_entity_names:
                    existing_entity_names.add(entity_name)

                doc_entities_data = {
                    "entity_names": list(existing_entity_names),
                    "count": len(existing_entity_names),
                    "update_time": int(time.time()),
                }

            # 存储更新的数据
            await self.lightrag.full_entities.upsert({doc_id: doc_entities_data})
            await self.lightrag.full_entities.index_done_callback()

            self.logger.debug(
                f"Added {len(entities_to_store)} multimodal main entities to full_entities for doc {doc_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Error storing multimodal entities to full_entities: {e}"
            )
            raise

    async def _batch_extract_entities_lightrag_style_type_aware(
        self, lightrag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """使用 LightRAG 的 extract_entities 进行批量实体关系提取"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import extract_entities

        # 获取流水线状态（与 LightRAG 一致）
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # 直接使用 LightRAG 的 extract_entities
        chunk_results = await extract_entities(
            chunks=lightrag_chunks,
            global_config=self.lightrag.__dict__,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        self.logger.info(
            f"Extracted entities from {len(lightrag_chunks)} multimodal chunks"
        )
        return chunk_results

    async def _batch_add_belongs_to_relations_type_aware(
        self, chunk_results: List[Tuple], multimodal_data_list: List[Dict[str, Any]]
    ) -> List[Tuple]:
        """为多模态实体添加 belongs_to 关系"""
        # 创建从 chunk_id 到 modal_entity_name 的映射
        chunk_to_modal_entity = {}
        chunk_to_file_path = {}

        for data in multimodal_data_list:
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # 使用与 _convert_to_lightrag_chunks_type_aware 中相同的模板格式化
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            chunk_to_modal_entity[chunk_id] = data["entity_info"]["entity_name"]
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")

        enhanced_chunk_results = []
        belongs_to_count = 0

        for maybe_nodes, maybe_edges in chunk_results:
            # 查找此块对应的 modal_entity_name
            chunk_id = None
            for nodes_dict in maybe_nodes.values():
                if nodes_dict:
                    chunk_id = nodes_dict[0].get("source_id")
                    break

            if chunk_id and chunk_id in chunk_to_modal_entity:
                modal_entity_name = chunk_to_modal_entity[chunk_id]
                file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")

                # 为所有提取的实体添加 belongs_to 关系
                for entity_name in maybe_nodes.keys():
                    if entity_name != modal_entity_name:  # 避免自身关系
                        belongs_to_relation = {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "source_id": chunk_id,
                            "weight": 10.0,
                            "file_path": file_path,
                        }

                        # 添加到 maybe_edges
                        edge_key = (entity_name, modal_entity_name)
                        if edge_key not in maybe_edges:
                            maybe_edges[edge_key] = []
                        maybe_edges[edge_key].append(belongs_to_relation)
                        belongs_to_count += 1

            enhanced_chunk_results.append((maybe_nodes, maybe_edges))

        self.logger.info(
            f"Added {belongs_to_count} belongs_to relations for multimodal entities"
        )
        return enhanced_chunk_results

    async def _batch_merge_lightrag_style_type_aware(
        self, enhanced_chunk_results: List[Tuple], file_path: str, doc_id: str = None
    ):
        """使用 LightRAG 的 merge_nodes_and_edges 进行批量合并"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import merge_nodes_and_edges

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        await merge_nodes_and_edges(
            chunk_results=enhanced_chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=self.lightrag.__dict__,
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            current_file_number=1,
            total_files=1,
            file_path=os.path.basename(file_path),
        )

        await self.lightrag._insert_done()

    async def _update_doc_status_with_chunks_type_aware(
        self, doc_id: str, chunk_ids: List[str]
    ):
        """使用多模态块更新文档状态"""
        try:
            # 获取当前文档状态
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

            if current_doc_status:
                existing_chunks_list = current_doc_status.get("chunks_list", [])
                existing_chunks_count = current_doc_status.get("chunks_count", 0)

                # 将多模态块添加到标准 chunks_list
                updated_chunks_list = existing_chunks_list + chunk_ids
                updated_chunks_count = existing_chunks_count + len(chunk_ids)

                # 使用集成的块列表更新文档状态
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,  # 保留现有字段
                            "chunks_list": updated_chunks_list,  # 集成的块列表
                            "chunks_count": updated_chunks_count,  # 更新的总计数
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )

                # 确保 doc_status 更新持久化到磁盘
                await self.lightrag.doc_status.index_done_callback()

                self.logger.info(
                    f"Updated doc_status: added {len(chunk_ids)} multimodal chunks to standard chunks_list "
                    f"(total chunks: {updated_chunks_count})"
                )

        except Exception as e:
            self.logger.warning(
                f"Error updating doc_status with multimodal chunks: {e}"
            )

    async def _mark_multimodal_processing_complete(self, doc_id: str):
        """在文档状态中标记多模态内容处理完成。"""
        try:
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,
                            "multimodal_processed": True,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )
                await self.lightrag.doc_status.index_done_callback()
                self.logger.debug(
                    f"Marked multimodal content processing as complete for document {doc_id}"
                )
        except Exception as e:
            self.logger.warning(
                f"Error marking multimodal processing as complete for document {doc_id}: {e}"
            )

    async def is_document_fully_processed(self, doc_id: str) -> bool:
        """
        检查文档是否已完全处理（文本和多模态内容）。

        Args:
            doc_id: 要检查的文档 ID

        Returns:
            bool: 如果文本和多模态内容都已处理，则返回 True
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return False

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)

            return text_processed and multimodal_processed

        except Exception as e:
            self.logger.error(
                f"Error checking document processing status for {doc_id}: {e}"
            )
            return False

    async def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        获取文档的详细处理状态。

        Args:
            doc_id: 要检查的文档 ID

        Returns:
            包含处理状态详情的字典
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return {
                    "exists": False,
                    "text_processed": False,
                    "multimodal_processed": False,
                    "fully_processed": False,
                    "chunks_count": 0,
                }

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)
            fully_processed = text_processed and multimodal_processed

            return {
                "exists": True,
                "text_processed": text_processed,
                "multimodal_processed": multimodal_processed,
                "fully_processed": fully_processed,
                "chunks_count": doc_status.get("chunks_count", 0),
                "chunks_list": doc_status.get("chunks_list", []),
                "status": doc_status.get("status", ""),
                "updated_at": doc_status.get("updated_at", ""),
                "raw_status": doc_status,
            }

        except Exception as e:
            self.logger.error(
                f"Error getting document processing status for {doc_id}: {e}"
            )
            return {
                "exists": False,
                "error": str(e),
                "text_processed": False,
                "multimodal_processed": False,
                "fully_processed": False,
                "chunks_count": 0,
            }

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        **kwargs,
    ):
        """
        完整的文档处理工作流

        Args:
            file_path: 要处理的文件路径
            output_dir: 输出目录（默认为 config.parser_output_dir）
            parse_method: 解析方法（默认为 config.parse_method）
            display_stats: 是否显示内容统计（默认为 config.display_content_stats）
            split_by_character: 可选的用于分割文本的字符
            split_by_character_only: 如果为 True，仅按指定字符分割
            doc_id: 可选的文档 ID，如果未提供将从内容生成
            **kwargs: 解析器的额外参数（例如：lang, device, start_page, end_page, formula, table, backend, source）
        """
        # 确保 LightRAG 已初始化
        await self._ensure_lightrag_initialized()

        # 如果未提供，使用配置默认值
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting complete document processing: {file_path}")

        # 步骤 1: 解析文档
        content_list, content_based_doc_id = await self.parse_document(
            file_path, output_dir, parse_method, display_stats, **kwargs
        )

        # 使用提供的 doc_id 或回退到基于内容的 doc_id
        if doc_id is None:
            doc_id = content_based_doc_id

        # 步骤 2: 分离文本和多模态内容
        text_content, multimodal_items = separate_content(content_list)

        # 步骤 2.5: 为多模态处理中的上下文提取设置内容源
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # 步骤 3: 使用所有参数插入纯文本内容
        if text_content.strip():
            file_name = os.path.basename(file_path)
            await insert_text_content(
                self.lightrag,
                input=text_content,
                file_paths=file_name,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )

        # 步骤 4: 处理多模态内容（使用专用处理器）
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_path, doc_id)
        else:
            # 如果没有多模态内容，标记多模态处理完成
            # 这确保文档状态正确反映所有处理的完成
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        self.logger.info(f"Document {file_path} processing complete!")

    async def process_document_complete_lightrag_api(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        scheme_name: str | None = None,
        parser: str | None = None,
        **kwargs,
    ):
        """
        专供 LightRAG 调用的 API: 完整的文档处理工作流

        Args:
            file_path: 要处理的文件路径
            output_dir: 输出目录（默认为 config.parser_output_dir）
            parse_method: 解析方法（默认为 config.parse_method）
            display_stats: 是否显示内容统计（默认为 config.display_content_stats）
            split_by_character: 可选的用于分割文本的字符
            split_by_character_only: 如果为 True，仅按指定字符分割
            doc_id: 可选的文档 ID，如果未提供将从内容生成
            **kwargs: 解析器的额外参数（例如：lang, device, start_page, end_page, formula, table, backend, source）
        """
        file_name = os.path.basename(file_path)
        doc_pre_id = f"doc-pre-{file_name}"
        pipeline_status = None
        pipeline_status_lock = None

        if parser:
            self.config.parser = parser

        current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)

        try:
            # 确保 LightRAG 已初始化
            result = await self._ensure_lightrag_initialized()
            if not result["success"]:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": result["error"],
                        }
                    }
                )
                return False

            # 如果未提供，使用配置默认值
            if output_dir is None:
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                parse_method = self.config.parse_method
            if display_stats is None:
                display_stats = self.config.display_content_stats

            self.logger.info(f"Starting complete document processing: {file_path}")

            # 初始化文档状态
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)
            if not current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            "status": DocStatus.READY,
                            "content": "",
                            "error_msg": "",
                            "content_summary": "",
                            "multimodal_content": [],
                            "scheme_name": scheme_name,
                            "content_length": 0,
                            "created_at": "",
                            "updated_at": "",
                            "file_path": file_name,
                        }
                    }
                )
                current_doc_status = await self.lightrag.doc_status.get_by_id(
                    doc_pre_id
                )

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            # 设置处理状态
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": True})
                pipeline_status["history_messages"].append("Now is not allowed to scan")

            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.HANDLING,
                        "error_msg": "",
                    }
                }
            )

            content_list = []
            content_based_doc_id = ""

            try:
                # 步骤 1: 解析文档
                content_list, content_based_doc_id = await self.parse_document(
                    file_path, output_dir, parse_method, display_stats, **kwargs
                )
            except MineruExecutionError as e:
                error_message = e.error_msg
                if isinstance(e.error_msg, list):
                    error_message = "\n".join(e.error_msg)
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": error_message,
                        }
                    }
                )
                self.logger.info(
                    f"Error processing document {file_path}: MineruExecutionError"
                )
                return False
            except Exception as e:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": str(e),
                        }
                    }
                )
                self.logger.info(f"Error processing document {file_path}: {str(e)}")
                return False

            # 使用提供的 doc_id 或回退到基于内容的 doc_id
            if doc_id is None:
                doc_id = content_based_doc_id

            # 步骤 2: 分离文本和多模态内容
            text_content, multimodal_items = separate_content(content_list)

            # 步骤 2.5: 为多模态处理中的上下文提取设置内容源
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # 步骤 3: 使用所有参数插入纯文本内容和多模态内容
            if text_content.strip():
                await insert_text_content_with_multimodal_content(
                    self.lightrag,
                    input=text_content,
                    multimodal_content=multimodal_items,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                    scheme_name=scheme_name,
                )

            self.logger.info(f"Document {file_path} processing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)

            # 将文档状态更新为失败
            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.FAILED,
                        "error_msg": str(e),
                    }
                }
            )
            await self.lightrag.doc_status.index_done_callback()

            # 更新流水线状态
            if pipeline_status_lock and pipeline_status:
                try:
                    async with pipeline_status_lock:
                        pipeline_status.update({"scan_disabled": False})
                        error_msg = (
                            f"RAGAnything processing failed for {file_name}: {str(e)}"
                        )
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)
                        pipeline_status["history_messages"].append(
                            "Now is allowed to scan"
                        )
                except Exception as pipeline_update_error:
                    self.logger.error(
                        f"Failed to update pipeline status: {pipeline_update_error}"
                    )

            return False

        finally:
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": False})
                pipeline_status["latest_message"] = (
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append(
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append("Now is allowed to scan")

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        display_stats: bool = None,
    ):
        """
        直接插入内容列表而不进行文档解析

        Args:
            content_list: 包含文本和多模态项的预解析内容列表。
                         每个项应该是具有以下结构的字典:
                         - 文本: {"type": "text", "text": "content", "page_idx": 0}
                         - 图像: {"type": "image", "img_path": "/absolute/path/to/image.jpg",
                                  "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}
                         - 表格: {"type": "table", "table_body": "markdown table",
                                  "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}
                         - 公式: {"type": "equation", "latex": "LaTeX formula",
                                     "text": "description", "page_idx": 3}
                         - 通用: {"type": "custom_type", "content": "any content", "page_idx": 4}
            file_path: 用于引用的参考文件路径/名称（默认为 "unknown_document"）
            split_by_character: 可选的用于分割文本的字符
            split_by_character_only: 如果为 True，仅按指定字符分割
            doc_id: 可选的文档 ID，如果未提供将从内容生成
            display_stats: 是否显示内容统计（默认为 config.display_content_stats）

        注意:
            - img_path 必须是图像文件的绝对路径
            - page_idx 表示内容出现的页码（从 0 开始索引）
            - 项按它们在列表中出现的顺序处理
        """
        # 确保 LightRAG 已初始化
        await self._ensure_lightrag_initialized()

        # 如果未提供，使用配置默认值
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(
            f"Starting direct content list insertion for: {file_path} ({len(content_list)} items)"
        )

        # 如果未提供，基于内容生成 doc_id
        if doc_id is None:
            doc_id = self._generate_content_based_doc_id(content_list)

        # 如果请求，显示内容统计
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # 按类型计数元素
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        # 步骤 1: 分离文本和多模态内容
        text_content, multimodal_items = separate_content(content_list)

        # 步骤 1.5: 为多模态处理中的上下文提取设置内容源
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # 步骤 2: 使用所有参数插入纯文本内容
        if text_content.strip():
            file_name = os.path.basename(file_path)
            await insert_text_content(
                self.lightrag,
                input=text_content,
                file_paths=file_name,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )

        # 步骤 3: 处理多模态内容（使用专用处理器）
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_path, doc_id)
        else:
            # 如果没有多模态内容，标记多模态处理完成
            # 这确保文档状态正确反映所有处理的完成
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        self.logger.info(f"Content list insertion complete for: {file_path}")
