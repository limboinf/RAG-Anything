"""
完整的文档解析与多模态内容写入流水线

该脚本整合：
1. 文档解析（使用可配置的解析器）
2. LightRAG 纯文本内容写入
3. 多模态内容的专项处理（通过不同的处理器）
"""

import os
from typing import Dict, Any, Optional, Callable
import sys
import asyncio
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 在导入 LightRAG 之前加载 .env 文件中的环境变量
# 这对离线环境中 TIKTOKEN_CACHE_DIR 的正常工作至关重要
# 操作系统的环境变量优先生效
load_dotenv(dotenv_path=".env", override=False)

from lightrag import LightRAG
from lightrag.utils import logger

# 导入配置与模块
from raganything.config import RAGAnythingConfig
from raganything.query import QueryMixin
from raganything.processor import ProcessorMixin
from raganything.batch import BatchMixin
from raganything.utils import get_processor_supports
from raganything.parser import MineruParser, DoclingParser

# 导入专用多模态处理器
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)


@dataclass
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
    """多模态文档处理流水线——提供完整的解析与写入能力"""

    # 核心组件
    # ---
    lightrag: Optional[LightRAG] = field(default=None)
    """可选的预先初始化 LightRAG 实例。"""

    llm_model_func: Optional[Callable] = field(default=None)
    """用于文本分析的 LLM 模型函数。"""

    vision_model_func: Optional[Callable] = field(default=None)
    """用于图像分析的视觉模型函数。"""

    embedding_func: Optional[Callable] = field(default=None)
    """用于文本向量化的嵌入函数。"""

    config: Optional[RAGAnythingConfig] = field(default=None)
    """配置对象；若未提供则依据环境变量创建。"""

    # LightRAG 配置
    # ---
    lightrag_kwargs: Dict[str, Any] = field(default_factory=dict)
    """在未传入 lightrag 时补充 LightRAG 初始化参数。
    可透传全部 LightRAG 配置项，例如：
    - kv_storage、vector_storage、graph_storage、doc_status_storage
    - top_k、chunk_top_k、max_entity_tokens、max_relation_tokens、max_total_tokens
    - cosine_threshold、related_chunk_number
    - chunk_token_size、chunk_overlap_token_size、tokenizer、tiktoken_model_name
    - embedding_batch_num、embedding_func_max_async、embedding_cache_config
    - llm_model_name、llm_model_max_token_size、llm_model_max_async、llm_model_kwargs
    - rerank_model_func、vector_db_storage_cls_kwargs、enable_llm_cache
    - max_parallel_insert、max_graph_nodes、addon_params 等。
    """

    # 内部状态
    # ---
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    """多模态处理器的字典集合。"""

    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    """为多模态处理器提供上下文内容的提取器。"""

    parse_cache: Optional[Any] = field(default=None, init=False)
    """利用 LightRAG KV 存储缓存解析结果。"""

    _parser_installation_checked: bool = field(default=False, init=False)
    """用于标记解析器安装状态是否已检查。"""

    def __post_init__(self):
        """遵循 LightRAG 模式的初始化收尾步骤"""
        # 如未传入配置则初始化默认配置
        if self.config is None:
            self.config = RAGAnythingConfig()

        # 设置工作目录
        self.working_dir = self.config.working_dir

        # 初始化日志记录器（复用现有 logger，避免重复配置）
        self.logger = logger

        # 初始化文档解析器
        self.doc_parser = (
            DoclingParser() if self.config.parser == "docling" else MineruParser()
        )

        # 注册清理方法
        atexit.register(self.close)

        # 如有需要则创建工作目录
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        # 输出配置详情
        self.logger.info("RAGAnything initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(
            f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
            f"Table: {self.config.enable_table_processing}, "
            f"Equation: {self.config.enable_equation_processing}"
        )
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")

    def close(self):
        """对象销毁时清理资源"""
        try:
            import asyncio

            if asyncio.get_event_loop().is_running():
                # 如处于异步上下文，则调度清理任务
                asyncio.create_task(self.finalize_storages())
            else:
                # 否则直接同步清理
                asyncio.run(self.finalize_storages())
        except Exception as e:
            # 日志器可能已被清理，此处改用 print
            print(f"Warning: Failed to finalize RAGAnything storages: {e}")

    def _create_context_config(self) -> ContextConfig:
        """依据 RAGAnything 配置生成上下文提取设置"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """使用 LightRAG 的 tokenizer 创建上下文提取器"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG must be initialized before creating context extractor"
            )

        context_config = self._create_context_config()
        return ContextExtractor(
            config=context_config, tokenizer=self.lightrag.tokenizer
        )

    def _initialize_processors(self):
        """结合所需模型函数初始化多模态处理器"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG instance must be initialized before creating processors"
            )

        # 创建上下文提取器
        self.context_extractor = self._create_context_extractor()

        # 依据配置生成不同类型的多模态处理器
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_equation_processing:
            self.modal_processors["equation"] = EquationModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        # 始终保留通用处理器作为兜底
        self.modal_processors["generic"] = GenericModalProcessor(
            lightrag=self.lightrag,
            modal_caption_func=self.llm_model_func,
            context_extractor=self.context_extractor,
        )

        self.logger.info("Multimodal processors initialized with context support")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")
        self.logger.info(f"Context configuration: {self._create_context_config()}")

    def update_config(self, **kwargs):
        """使用新值更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    async def _ensure_lightrag_initialized(self):
        """确保 LightRAG 实例已就绪，必要时自动创建"""
        try:
            # 优先确认解析器是否安装
            if not self._parser_installation_checked:
                if not self.doc_parser.check_installation():
                    error_msg = (
                        f"Parser '{self.config.parser}' is not properly installed. "
                        "Please install it using 'pip install' or 'uv pip install'."
                    )
                    self.logger.error(error_msg)
                    return {"success": False, "error": error_msg}

                self._parser_installation_checked = True
                self.logger.info(f"Parser '{self.config.parser}' installation verified")

            if self.lightrag is not None:
                # 当 LightRAG 由外部传入时，也需确认其状态完整
                try:
                    # 确认 LightRAG 各类存储已完成初始化
                    if (
                        not hasattr(self.lightrag, "_storages_status")
                        or self.lightrag._storages_status.name != "INITIALIZED"
                    ):
                        self.logger.info(
                            "Initializing storages for pre-provided LightRAG instance"
                        )
                        await self.lightrag.initialize_storages()
                        from lightrag.kg.shared_storage import (
                            initialize_pipeline_status,
                        )

                        await initialize_pipeline_status()

                    # 初始化解析缓存
                    if self.parse_cache is None:
                        self.logger.info(
                            "Initializing parse cache for pre-provided LightRAG instance"
                        )
                        self.parse_cache = (
                            self.lightrag.key_string_value_json_storage_cls(
                                namespace="parse_cache",
                                workspace=self.lightrag.workspace,
                                global_config=self.lightrag.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.parse_cache.initialize()

                    # 如未创建处理器则进行初始化
                    if not self.modal_processors:
                        self._initialize_processors()

                    return {"success": True}

                except Exception as e:
                    error_msg = (
                        f"Failed to initialize pre-provided LightRAG instance: {str(e)}"
                    )
                    self.logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg}

            # 创建全新 LightRAG 时校验必备函数
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                error_msg = "embedding_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            from lightrag.kg.shared_storage import initialize_pipeline_status

            # 准备 LightRAG 初始化参数
            lightrag_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }

            # 合并用户传入的 lightrag_kwargs 以覆写默认值
            lightrag_params.update(self.lightrag_kwargs)

            # 记录用于初始化的参数（排除敏感字段）
            log_params = {
                k: v
                for k, v in lightrag_params.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(f"Initializing LightRAG with parameters: {log_params}")

            try:
                # 使用合并参数创建 LightRAG 实例
                self.lightrag = LightRAG(**lightrag_params)
                await self.lightrag.initialize_storages()
                await initialize_pipeline_status()

                # 基于 LightRAG 的 KV 存储创建解析缓存
                self.parse_cache = self.lightrag.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.lightrag.workspace,
                    global_config=self.lightrag.__dict__,
                    embedding_func=self.embedding_func,
                )
                await self.parse_cache.initialize()

                # LightRAG 就绪后再初始化处理器
                self._initialize_processors()

                self.logger.info(
                    "LightRAG, parse cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                error_msg = f"Failed to initialize LightRAG instance: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during LightRAG initialization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}

    async def finalize_storages(self):
        """完成解析缓存及 LightRAG 存储的收尾工作

        关闭系统时应调用本方法以确保资源正确释放并持久化缓存数据。
        该过程会同时处理解析缓存与 LightRAG 内部各类存储。

        示例：
            try:
                rag_anything = RAGAnything(...)
                await rag_anything.process_file("document.pdf")
                # ... 其他操作 ...
            finally:
                # 始终记得清理存储
                if rag_anything:
                    await rag_anything.finalize_storages()

        说明：
            - __del__ 中会自动调用，但生产环境建议显式执行
            - 所有清理任务并发进行以提升效率
        """
        try:
            tasks = []

            # 如果存在解析缓存则加入清理任务
            if self.parse_cache is not None:
                tasks.append(self.parse_cache.finalize())
                self.logger.debug("Scheduled parse cache finalization")

            # 若 LightRAG 已初始化则清理其存储
            if self.lightrag is not None:
                tasks.append(self.lightrag.finalize_storages())
                self.logger.debug("Scheduled LightRAG storages finalization")

            # 并发执行所有清理任务
            if tasks:
                await asyncio.gather(*tasks)
                self.logger.info("Successfully finalized all RAGAnything storages")
            else:
                self.logger.debug("No storages to finalize")

        except Exception as e:
            self.logger.error(f"Error during storage finalization: {e}")
            raise

    def check_parser_installation(self) -> bool:
        """
        检查当前配置的解析器是否正确安装

        Returns:
            bool: 若解析器已正确安装则返回 True
        """
        return self.doc_parser.check_installation()

    def verify_parser_installation_once(self) -> bool:
        if not self._parser_installation_checked:
            if not self.doc_parser.check_installation():
                raise RuntimeError(
                    f"Parser '{self.config.parser}' is not properly installed. "
                    "Please install it using pip install or uv pip install."
                )
            self._parser_installation_checked = True
            self.logger.info(f"Parser '{self.config.parser}' installation verified")
        return True

    def get_config_info(self) -> Dict[str, Any]:
        """获取当前配置详情"""
        config_info = {
            "directory": {
                "working_dir": self.config.working_dir,
                "parser_output_dir": self.config.parser_output_dir,
            },
            "parsing": {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
                "display_content_stats": self.config.display_content_stats,
            },
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
            "logging": {
                "note": "Logging fields have been removed - configure logging externally",
            },
        }

        # 如存在自定义 LightRAG 配置则附加展示
        if self.lightrag_kwargs:
            # 过滤掉可调用对象与敏感字段再展示
            safe_kwargs = {
                k: v
                for k, v in self.lightrag_kwargs.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            config_info["lightrag_config"] = {
                "custom_parameters": safe_kwargs,
                "note": "LightRAG will be initialized with these additional parameters",
            }
        else:
            config_info["lightrag_config"] = {
                "custom_parameters": {},
                "note": "Using default LightRAG parameters",
            }

        return config_info

    def set_content_source_for_context(
        self, content_source, content_format: str = "auto"
    ):
        """为所有多模态处理器设置上下文内容来源

        Args:
            content_source: 上下文提取所依赖的内容源（例如 MinerU 内容列表）
            content_format: 内容源格式（"minerU"、"text_chunks"、"auto"）
        """
        if not self.modal_processors:
            self.logger.warning(
                "Modal processors not initialized. Content source will be set when processors are created."
            )
            return

        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"Set content source for {processor_name} processor")
            except Exception as e:
                self.logger.error(
                    f"Failed to set content source for {processor_name}: {e}"
                )

        self.logger.info(
            f"Content source set for context extraction (format: {content_format})"
        )

    def update_context_config(self, **context_kwargs):
        """更新上下文提取相关配置

        Args:
            **context_kwargs: 需要变更的上下文配置项
                （如 context_window、context_mode、max_context_tokens 等）
        """
        # Update the main config
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated context config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown context config parameter: {key}")

        # 若处理器已初始化则根据新配置重建上下文提取器
        if self.lightrag and self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                # 将新的上下文提取器分发给所有处理器
                for processor_name, processor in self.modal_processors.items():
                    processor.context_extractor = self.context_extractor

                self.logger.info(
                    "Context configuration updated and applied to all processors"
                )
                self.logger.info(
                    f"New context configuration: {self._create_context_config()}"
                )
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> Dict[str, Any]:
        """获取多模态处理器的状态信息"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedding_func
                else "Not provided",
            },
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
            base_info["processors"] = {}
        else:
            base_info["status"] = "Initialized"
            base_info["processors"] = {}

            for proc_type, processor in self.modal_processors.items():
                base_info["processors"][proc_type] = {
                    "class": processor.__class__.__name__,
                    "supports": get_processor_supports(proc_type),
                    "enabled": True,
                }

        return base_info
