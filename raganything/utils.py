"""
RAGAnything 实用工具函数

包含用于内容分离、文本插入和其他实用工具的辅助函数
"""

import base64
from typing import Dict, List, Any, Tuple
from pathlib import Path
from lightrag.utils import logger


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    分离文本内容和多模态内容

    Args:
        content_list: 来自 MinerU 解析的内容列表

    Returns:
        (text_content, multimodal_items): 纯文本内容和多模态项目列表
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # 文本内容
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # 多模态内容（图像、表格、公式等）
            multimodal_items.append(item)

    # 合并所有文本内容
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # 统计多模态类型
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    将图像文件编码为 base64 字符串

    Args:
        image_path: 图像文件的路径

    Returns:
        str: Base64 编码的字符串，如果编码失败则返回空字符串
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    验证文件是否为有效的图像文件

    Args:
        image_path: 图像文件的路径
        max_size_mb: 最大文件大小（MB）

    Returns:
        bool: 如果有效则返回 True，否则返回 False
    """
    try:
        path = Path(image_path)

        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # 检查文件是否存在
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        # 检查文件扩展名
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # 检查文件大小
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


async def insert_text_content(
    lightrag,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
):
    """
    将纯文本内容插入 LightRAG

    Args:
        lightrag: LightRAG 实例
        input: 单个文档字符串或文档字符串列表
        split_by_character: 如果 split_by_character 不为 None，按字符分割字符串，如果块长度超过
        chunk_token_size，将再次按 token 大小分割。
        split_by_character_only: 如果 split_by_character_only 为 True，仅按字符分割字符串，当
        split_by_character 为 None 时，此参数被忽略。
        ids: 文档 ID 的单个字符串或唯一文档 ID 列表，如果未提供，将生成 MD5 哈希 ID
        file_paths: 文件路径的单个字符串或文件路径列表，用于引用
    """
    logger.info("Starting text content insertion into LightRAG...")

    # 使用所有参数调用 LightRAG 的插入方法
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    lightrag,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    将纯文本内容插入 LightRAG

    Args:
        lightrag: LightRAG 实例
        input: 单个文档字符串或文档字符串列表
        multimodal_content: 多模态内容列表（可选）
        split_by_character: 如果 split_by_character 不为 None，按字符分割字符串，如果块长度超过
        chunk_token_size，将再次按 token 大小分割。
        split_by_character_only: 如果 split_by_character_only 为 True，仅按字符分割字符串，当
        split_by_character 为 None 时，此参数被忽略。
        ids: 文档 ID 的单个字符串或唯一文档 ID 列表，如果未提供，将生成 MD5 哈希 ID
        file_paths: 文件路径的单个字符串或文件路径列表，用于引用
        scheme_name: 方案名称（可选）
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    try:
        await lightrag.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    根据内容类型获取适当的处理器

    Args:
        modal_processors: 可用处理器字典
        content_type: 内容类型

    Returns:
        对应的处理器实例
    """
    # 直接映射到相应的处理器
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # 对于其他类型，使用通用处理器
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """获取处理器支持的功能"""
    supports_map = {
        "image": [
            "图像内容分析",
            "视觉理解",
            "图像描述生成",
            "图像实体提取",
        ],
        "table": [
            "表格结构分析",
            "数据统计",
            "趋势识别",
            "表格实体提取",
        ],
        "equation": [
            "数学公式解析",
            "变量识别",
            "公式含义解释",
            "公式实体提取",
        ],
        "generic": [
            "通用内容分析",
            "结构化处理",
            "实体提取",
        ],
    }
    return supports_map.get(proc_type, ["基本处理"])
