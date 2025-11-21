"""
RAGAnything 配置类

包含支持环境变量的配置数据类
"""

from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value


@dataclass
class RAGAnythingConfig:
    """支持环境变量的 RAGAnything 配置类"""

    # 目录配置
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", "./rag_storage", str))
    """存储 RAG 存储和缓存文件的目录。"""

    # 解析器配置
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))
    """文档解析的默认解析方法：'auto'、'ocr' 或 'txt'。"""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", "./output", str))
    """解析内容的默认输出目录。"""

    parser: str = field(default=get_env_value("PARSER", "mineru", str))
    """解析器选择：'mineru' 或 'docling'。"""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", True, bool)
    )
    """解析过程中是否显示内容统计信息。"""

    # 多模态处理配置
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )
    """启用图像内容处理。"""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )
    """启用表格内容处理。"""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool)
    )
    """启用公式内容处理。"""

    # 批处理配置
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", 1, int)
    )
    """并发处理的最大文件数量。"""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: get_env_value(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
            str,
        ).split(",")
    )
    """批处理支持的文件扩展名列表。"""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool)
    )
    """批处理模式下是否递归处理子文件夹。"""

    # 上下文提取配置
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))
    """在当前项目前后包含的页面/块数量，用于提供上下文。"""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))
    """上下文提取模式：'page' 表示基于页面，'chunk' 表示基于块。"""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 2000, int)
    )
    """提取的上下文中的最大 token 数量。"""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", True, bool))
    """上下文中是否包含文档标题和标头。"""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )
    """上下文中是否包含图像/表格标题。"""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "CONTEXT_FILTER_CONTENT_TYPES", "text", str
        ).split(",")
    )
    """上下文提取中包含的内容类型（例如 'text'、'image'、'table'）。"""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", "minerU", str))
    """处理文档时上下文提取的默认内容格式。"""

    def __post_init__(self):
        """向后兼容性的初始化后设置"""
        # 支持旧版环境变量名以保持向后兼容性
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def mineru_parse_method(self) -> str:
        """
        旧代码的向后兼容性属性。

        .. deprecated::
           请使用 `parse_method` 替代。此属性将在未来版本中移除。
        """
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """向后兼容性的 setter"""
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value
