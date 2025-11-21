"""
RAGAnything 查询功能

包含文本和多模态查询的所有查询相关方法
"""

import json
import hashlib
import re
from typing import Dict, List, Any
from pathlib import Path
from lightrag import QueryParam
from lightrag.utils import always_get_an_event_loop
from raganything.prompt import PROMPTS
from raganything.utils import (
    get_processor_for_type,
    encode_image_to_base64,
    validate_image_file,
)


class QueryMixin:
    """包含 RAGAnything 查询功能的 QueryMixin 类"""

    def _generate_multimodal_cache_key(
        self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
    ) -> str:
        """
        为多模态查询生成缓存键

        Args:
            query: 基础查询文本
            multimodal_content: 多模态内容列表
            mode: 查询模式
            **kwargs: 额外参数

        Returns:
            str: 缓存键哈希值
        """
        # 创建查询参数的标准化表示
        cache_data = {
            "query": query.strip(),
            "mode": mode,
        }

        # 标准化多模态内容以实现稳定的缓存
        normalized_content = []
        if multimodal_content:
            for item in multimodal_content:
                if isinstance(item, dict):
                    normalized_item = {}
                    for key, value in item.items():
                        # 对于文件路径，使用基础名称使缓存更具可移植性
                        if key in [
                            "img_path",
                            "image_path",
                            "file_path",
                        ] and isinstance(value, str):
                            normalized_item[key] = Path(value).name
                        # 对于大型内容，创建哈希值而不是直接存储
                        elif (
                            key in ["table_data", "table_body"]
                            and isinstance(value, str)
                            and len(value) > 200
                        ):
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        else:
                            normalized_item[key] = value
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)

        cache_data["multimodal_content"] = normalized_content

        # 将相关的 kwargs 添加到缓存数据
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "stream",
                "response_type",
                "top_k",
                "max_tokens",
                "temperature",
                # "only_need_context",
                # "only_need_prompt",
            ]
        }
        cache_data.update(relevant_kwargs)

        # 从缓存数据生成哈希值
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return f"multimodal_query:{cache_hash}"

    async def aquery(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        纯文本查询 - 直接调用 LightRAG 的查询功能

        Args:
            query: 查询文本
            mode: 查询模式（"local"、"global"、"hybrid"、"naive"、"mix"、"bypass"）
            **kwargs: 其他查询参数，将传递给 QueryParam
                - vlm_enhanced: bool，当 vision_model_func 可用时默认为 True。
                  如果为 True，将解析检索上下文中的图像路径并将其替换为
                  base64 编码的图像以供 VLM 处理。

        Returns:
            str: 查询结果
        """
        if self.lightrag is None:
            raise ValueError(
                "No LightRAG instance available. Please process documents first or provide a pre-initialized LightRAG instance."
            )

        # 检查是否应使用 VLM 增强查询
        vlm_enhanced = kwargs.pop("vlm_enhanced", None)

        # 根据可用性自动确定 VLM 增强
        if vlm_enhanced is None:
            vlm_enhanced = (
                hasattr(self, "vision_model_func")
                and self.vision_model_func is not None
            )

        # 如果启用且可用，使用 VLM 增强查询
        if (
            vlm_enhanced
            and hasattr(self, "vision_model_func")
            and self.vision_model_func
        ):
            return await self.aquery_vlm_enhanced(query, mode=mode, **kwargs)
        elif vlm_enhanced and (
            not hasattr(self, "vision_model_func") or not self.vision_model_func
        ):
            self.logger.warning(
                "VLM enhanced query requested but vision_model_func is not available, falling back to normal query"
            )

        # 创建查询参数
        query_param = QueryParam(mode=mode, **kwargs)

        self.logger.info(f"Executing text query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # 调用 LightRAG 的查询方法
        result = await self.lightrag.aquery(query, param=query_param)

        self.logger.info("Text query completed")
        return result

    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        多模态查询 - 结合文本和多模态内容进行查询

        Args:
            query: 基础查询文本
            multimodal_content: 多模态内容列表，每个元素包含：
                - type: 内容类型（"image"、"table"、"equation" 等）
                - 其他字段取决于类型（例如 img_path、table_data、latex 等）
            mode: 查询模式（"local"、"global"、"hybrid"、"naive"、"mix"、"bypass"）
            **kwargs: 其他查询参数，将传递给 QueryParam

        Returns:
            str: 查询结果

        Examples:
            # 纯文本查询
            result = await rag.query_with_multimodal("What is machine learning?")

            # 图像查询
            result = await rag.query_with_multimodal(
                "Analyze the content in this image",
                multimodal_content=[{
                    "type": "image",
                    "img_path": "./image.jpg"
                }]
            )

            # 表格查询
            result = await rag.query_with_multimodal(
                "Analyze the data trends in this table",
                multimodal_content=[{
                    "type": "table",
                    "table_data": "Name,Age\nAlice,25\nBob,30"
                }]
            )
        """
        # 确保 LightRAG 已初始化
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Executing multimodal query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # 如果没有多模态内容，回退到纯文本查询
        if not multimodal_content:
            self.logger.info("No multimodal content provided, executing text query")
            return await self.aquery(query, mode=mode, **kwargs)

        # 为多模态查询生成缓存键
        cache_key = self._generate_multimodal_cache_key(
            query, multimodal_content, mode, **kwargs
        )

        # 如果可用且已启用，检查缓存
        cached_result = None
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    cached_result = await self.lightrag.llm_response_cache.get_by_id(
                        cache_key
                    )
                    if cached_result and isinstance(cached_result, dict):
                        result_content = cached_result.get("return")
                        if result_content:
                            self.logger.info(
                                f"Multimodal query cache hit: {cache_key[:16]}..."
                            )
                            return result_content
                except Exception as e:
                    self.logger.debug(f"Error accessing multimodal query cache: {e}")

        # 处理多模态内容以生成增强查询文本
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )

        self.logger.info(
            f"Generated enhanced query length: {len(enhanced_query)} characters"
        )

        # 执行增强查询
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)

        # 如果可用且已启用，保存到缓存
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    # Create cache entry for multimodal query
                    cache_entry = {
                        "return": result,
                        "cache_type": "multimodal_query",
                        "original_query": query,
                        "multimodal_content_count": len(multimodal_content),
                        "mode": mode,
                    }

                    await self.lightrag.llm_response_cache.upsert(
                        {cache_key: cache_entry}
                    )
                    self.logger.info(
                        f"Saved multimodal query result to cache: {cache_key[:16]}..."
                    )
                except Exception as e:
                    self.logger.debug(f"Error saving multimodal query to cache: {e}")

        # 确保缓存持久化到磁盘
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            try:
                await self.lightrag.llm_response_cache.index_done_callback()
            except Exception as e:
                self.logger.debug(f"Error persisting multimodal query cache: {e}")

        self.logger.info("Multimodal query completed")
        return result

    async def aquery_vlm_enhanced(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        VLM 增强查询 - 将检索上下文中的图像路径替换为 base64 编码的图像以供 VLM 处理

        Args:
            query: 用户查询
            mode: 底层 LightRAG 查询模式
            **kwargs: 其他查询参数

        Returns:
            str: VLM 查询结果
        """
        # 确保 VLM 可用
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "VLM enhanced query requires vision_model_func. "
                "Please provide a vision model function when initializing RAGAnything."
            )

        # 确保 LightRAG 已初始化
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

        # 清除之前的图像缓存
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # 1. 获取原始检索提示（不生成最终答案）
        query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        raw_prompt = await self.lightrag.aquery(query, param=query_param)

        self.logger.debug("Retrieved raw prompt from LightRAG")

        # 2. 提取和处理图像路径
        enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
            raw_prompt
        )

        if not images_found:
            self.logger.info("No valid images found, falling back to normal query")
            # Fallback to normal query
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(query, param=query_param)

        self.logger.info(f"Processed {images_found} images for VLM")

        # 3. Build VLM message format
        messages = self._build_vlm_messages_with_images(enhanced_prompt, query)

        # 4. Call VLM for question answering
        result = await self._call_vlm_with_multimodal_content(messages)

        self.logger.info("VLM enhanced query completed")
        return result

    async def _process_multimodal_query_content(
        self, base_query: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """
        处理多模态查询内容以生成增强查询文本

        Args:
            base_query: 基础查询文本
            multimodal_content: 多模态内容列表

        Returns:
            str: 增强查询文本
        """
        self.logger.info("开始处理多模态查询内容...")

        enhanced_parts = [f"用户查询: {base_query}"]

        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            self.logger.info(
                f"正在处理第 {i+1}/{len(multimodal_content)} 个多模态内容: {content_type}"
            )

            try:
                # 获取适当的处理器
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # 生成内容描述
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\n相关的 {content_type} 内容: {description}"
                    )
                else:
                    # 如果没有适当的处理器,使用基本描述
                    basic_desc = str(content)[:200]
                    enhanced_parts.append(
                        f"\n相关的 {content_type} 内容: {basic_desc}"
                    )

            except Exception as e:
                self.logger.error(f"处理多模态内容时出错: {str(e)}")
                # 继续处理其他内容
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]

        self.logger.info("多模态查询内容处理完成")
        return enhanced_query

    async def _generate_query_content_description(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """
        为查询生成内容描述

        Args:
            processor: 多模态处理器
            content: 内容数据
            content_type: 内容类型

        Returns:
            str: 内容描述
        """
        try:
            if content_type == "image":
                return await self._describe_image_for_query(processor, content)
            elif content_type == "table":
                return await self._describe_table_for_query(processor, content)
            elif content_type == "equation":
                return await self._describe_equation_for_query(processor, content)
            else:
                return await self._describe_generic_for_query(
                    processor, content, content_type
                )

        except Exception as e:
            self.logger.error(f"Error generating {content_type} description: {str(e)}")
            return f"{content_type} content: {str(content)[:100]}"

    async def _describe_image_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """为查询生成图像描述"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        if image_path and Path(image_path).exists():
            # 如果图像存在,使用视觉模型生成描述
            image_base64 = processor._encode_image_to_base64(image_path)
            if image_base64:
                prompt = PROMPTS["QUERY_IMAGE_DESCRIPTION"]
                description = await processor.modal_caption_func(
                    prompt,
                    image_data=image_base64,
                    system_prompt=PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
                )
                return description

        # 如果图像不存在或处理失败,使用现有信息
        parts = []
        if image_path:
            parts.append(f"图像路径: {image_path}")
        if captions:
            parts.append(f"图像标题: {', '.join(captions)}")
        if footnotes:
            parts.append(f"图像注释: {', '.join(footnotes)}")

        return "; ".join(parts) if parts else "图像内容信息不完整"

    async def _describe_table_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """为查询生成表格描述"""
        table_data = content.get("table_data", "")
        table_caption = content.get("table_caption", "")

        prompt = PROMPTS["QUERY_TABLE_ANALYSIS"].format(
            table_data=table_data, table_caption=table_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_equation_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """为查询生成公式描述"""
        latex = content.get("latex", "")
        equation_caption = content.get("equation_caption", "")

        prompt = PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
            latex=latex, equation_caption=equation_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_generic_for_query(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """为查询生成通用内容描述"""
        content_str = str(content)

        prompt = PROMPTS["QUERY_GENERIC_ANALYSIS"].format(
            content_type=content_type, content_str=content_str
        )

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"].format(
                content_type=content_type
            ),
        )

        return description

    async def _process_image_paths_for_vlm(self, prompt: str) -> tuple[str, int]:
        """
        处理提示中的图像路径,保留原始路径并添加 VLM 标记

        Args:
            prompt: 原始提示

        Returns:
            tuple: (处理后的提示, 图像数量)
        """
        enhanced_prompt = prompt
        images_processed = 0

        # 初始化图像缓存
        self._current_images_base64 = []

        # 用于匹配图像路径的增强正则表达式模式
        # 仅匹配以图像文件扩展名结尾的路径
        image_path_pattern = (
            r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
        )

        # 首先,让我们看看找到了什么匹配项
        matches = re.findall(image_path_pattern, prompt)
        self.logger.info(f"在提示中找到 {len(matches)} 个图像路径匹配项")

        def replace_image_path(match):
            nonlocal images_processed

            image_path = match.group(1).strip()
            self.logger.debug(f"正在处理图像路径: '{image_path}'")

            # 验证路径格式(基本检查)
            if not image_path or len(image_path) < 3:
                self.logger.warning(f"无效的图像路径格式: {image_path}")
                return match.group(0)  # 保留原始内容

            # 使用工具函数验证图像文件
            self.logger.debug(f"正在调用 validate_image_file 验证: {image_path}")
            is_valid = validate_image_file(image_path)
            self.logger.debug(f"{image_path} 的验证结果: {is_valid}")

            if not is_valid:
                self.logger.warning(f"图像验证失败: {image_path}")
                return match.group(0)  # 如果验证失败,保留原始内容

            try:
                # 使用工具函数将图像编码为 base64
                self.logger.debug(f"尝试编码图像: {image_path}")
                image_base64 = encode_image_to_base64(image_path)
                if image_base64:
                    images_processed += 1
                    # 将 base64 保存到实例变量以供后续使用
                    self._current_images_base64.append(image_base64)

                    # 保留原始路径信息并添加 VLM 标记
                    result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
                    self.logger.debug(
                        f"成功处理图像 {images_processed}: {image_path}"
                    )
                    return result
                else:
                    self.logger.error(f"图像编码失败: {image_path}")
                    return match.group(0)  # 如果编码失败,保留原始内容

            except Exception as e:
                self.logger.error(f"处理图像 {image_path} 失败: {e}")
                return match.group(0)  # 保留原始内容

        # 执行替换
        enhanced_prompt = re.sub(
            image_path_pattern, replace_image_path, enhanced_prompt
        )

        return enhanced_prompt, images_processed

    def _build_vlm_messages_with_images(
        self, enhanced_prompt: str, user_query: str
    ) -> List[Dict]:
        """
        构建 VLM 消息格式,使用标记将图像与文本位置对应

        Args:
            enhanced_prompt: 带有图像标记的增强提示
            user_query: 用户查询

        Returns:
            List[Dict]: VLM 消息格式
        """
        images_base64 = getattr(self, "_current_images_base64", [])

        if not images_base64:
            # 纯文本模式
            return [
                {
                    "role": "user",
                    "content": f"上下文:\n{enhanced_prompt}\n\n用户问题: {user_query}",
                }
            ]

        # 构建多模态内容
        content_parts = []

        # 在图像标记处分割文本并插入图像
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        for i, text_part in enumerate(text_parts):
            if i == 0:
                # 第一个文本部分
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                # 查找标记编号并插入对应的图像
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = (
                        int(marker_match.group(1)) - 1
                    )  # 转换为从 0 开始的索引
                    remaining_text = marker_match.group(2)

                    # 插入对应的图像
                    if 0 <= image_num < len(images_base64):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                                },
                            }
                        )

                    # 插入剩余文本
                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})

        # 添加用户问题
        content_parts.append(
            {
                "type": "text",
                "text": f"\n\n用户问题: {user_query}\n\n请根据提供的上下文和图像进行回答。",
            }
        )

        return [
            {
                "role": "system",
                "content": "你是一个有帮助的助手,可以分析文本和图像内容以提供全面的答案。",
            },
            {"role": "user", "content": content_parts},
        ]

    async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
        """
        调用 VLM 处理多模态内容

        Args:
            messages: VLM 消息格式

        Returns:
            str: VLM 响应结果
        """
        try:
            user_message = messages[1]
            content = user_message["content"]
            system_prompt = messages[0]["content"]

            if isinstance(content, str):
                # 纯文本模式
                result = await self.vision_model_func(
                    content, system_prompt=system_prompt
                )
            else:
                # 多模态模式 - 直接将完整消息传递给 VLM
                result = await self.vision_model_func(
                    "",  # 空提示,因为我们使用的是消息格式
                    messages=messages,
                )

            return result

        except Exception as e:
            self.logger.error(f"VLM 调用失败: {e}")
            raise

    # 同步版本的查询方法
    def query(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        纯文本查询的同步版本

        Args:
            query: 查询文本
            mode: 查询模式（"local"、"global"、"hybrid"、"naive"、"mix"、"bypass"）
            **kwargs: 其他查询参数,将传递给 QueryParam
                - vlm_enhanced: bool,当 vision_model_func 可用时默认为 True。
                  如果为 True,将解析检索上下文中的图像路径并将其替换为
                  base64 编码的图像以供 VLM 处理。

        Returns:
            str: 查询结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

    def query_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        多模态查询的同步版本

        Args:
            query: 基础查询文本
            multimodal_content: 多模态内容列表,每个元素包含:
                - type: 内容类型（"image"、"table"、"equation" 等）
                - 其他字段取决于类型（例如 img_path、table_data、latex 等）
            mode: 查询模式（"local"、"global"、"hybrid"、"naive"、"mix"、"bypass"）
            **kwargs: 其他查询参数,将传递给 QueryParam

        Returns:
            str: 查询结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )
