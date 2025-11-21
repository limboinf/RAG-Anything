#!/usr/bin/env python
"""
演示使用 RAGAnything 直接插入内容列表的示例脚本

此示例展示如何：
1. 创建包含不同内容类型的简单内容列表
2. 使用 insert_content_list() 方法直接插入内容列表而无需文档解析
3. 使用 aquery() 方法执行纯文本查询
4. 使用 aquery_with_multimodal() 方法对特定多模态内容执行多模态查询
5. 处理插入知识库中的不同类型多模态内容
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# 将项目根目录添加到 Python 路径
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """配置应用程序的日志记录"""
    # 从环境变量获取日志目录路径或使用当前目录
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "insert_content_list_example.log")
    )

    print(f"\nInsert Content List example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # 从环境变量获取日志文件最大大小和备份数量
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 默认 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # 默认 5 个备份

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # 将日志记录器级别设置为 INFO
    logger.setLevel(logging.INFO)
    # 如果需要，启用详细调试
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


def create_sample_content_list():
    """
    创建用于测试 insert_content_list 功能的简单内容列表

    返回:
        List[Dict]: 包含各种内容类型的示例内容列表

    注意:
        - img_path 应为图像文件的绝对路径
        - page_idx 表示内容出现的页码（从 0 开始）
    """
    content_list = [
        # 引言文本
        {
            "type": "text",
            "text": "Welcome to the RAGAnything System Documentation. This guide covers the advanced multimodal document processing capabilities and features of our comprehensive RAG system.",
            "page_idx": 0,  # 此内容出现的页码
        },
        # 系统架构图
        {
            "type": "image",
            "img_path": "/absolute/path/to/system_architecture.jpg",  # 重要：使用图像文件的绝对路径
            "image_caption": ["Figure 1: RAGAnything System Architecture"],
            "image_footnote": [
                "The architecture shows the complete pipeline from document parsing to multimodal query processing"
            ],
            "page_idx": 1,  # 此图像出现的页码
        },
        # 性能比较表格
        {
            "type": "table",
            "table_body": """| System | Accuracy | Processing Speed | Memory Usage |
                            |--------|----------|------------------|--------------|
                            | RAGAnything | 95.2% | 120ms | 2.1GB |
                            | Traditional RAG | 87.3% | 180ms | 3.2GB |
                            | Baseline System | 82.1% | 220ms | 4.1GB |
                            | Simple Retrieval | 76.5% | 95ms | 1.8GB |""",
            "table_caption": [
                "Table 1: Performance Comparison of Different RAG Systems"
            ],
            "table_footnote": [
                "All tests conducted on the same hardware with identical test datasets"
            ],
            "page_idx": 2,  # 此表格出现的页码
        },
        # 数学公式
        {
            "type": "equation",
            "latex": "Relevance(d, q) = \\sum_{i=1}^{n} w_i \\cdot sim(t_i^d, t_i^q) \\cdot \\alpha_i",
            "text": "Document relevance scoring formula where w_i are term weights, sim() is similarity function, and α_i are modality importance factors",
            "page_idx": 3,  # 此方程出现的页码
        },
        # 功能描述
        {
            "type": "text",
            "text": "The system supports multiple content modalities including text, images, tables, and mathematical equations. Each modality is processed using specialized processors optimized for that content type.",
            "page_idx": 4,  # 此内容出现的页码
        },
        # 技术规格表格
        {
            "type": "table",
            "table_body": """| Feature | Specification |
                            |---------|---------------|
                            | Supported Formats | PDF, DOCX, PPTX, XLSX, Images |
                            | Max Document Size | 100MB |
                            | Concurrent Processing | Up to 8 documents |
                            | Query Response Time | <200ms average |
                            | Knowledge Graph Nodes | Up to 1M entities |""",
            "table_caption": ["Table 2: Technical Specifications"],
            "table_footnote": [
                "Specifications may vary based on hardware configuration"
            ],
            "page_idx": 5,  # 此表格出现的页码
        },
        # 结论
        {
            "type": "text",
            "text": "RAGAnything represents a significant advancement in multimodal document processing, providing comprehensive solutions for complex knowledge extraction and retrieval tasks.",
            "page_idx": 6,  # 此内容出现的页码
        },
    ]

    return content_list


async def demo_insert_content_list(
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
):
    """
    演示使用 RAGAnything 插入和查询内容列表

    参数:
        api_key: OpenAI API 密钥
        base_url: API 的可选基础 URL
        working_dir: RAG 存储的工作目录
    """
    try:
        # 创建 RAGAnything 配置
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            display_content_stats=True,  # 显示内容统计
        )

        # 定义 LLM 模型函数
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # 定义用于图像处理的视觉模型函数
        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs
        ):
            if image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # 定义嵌入函数 - 使用环境变量进行配置
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
            ),
        )

        # 初始化 RAGAnything
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # 创建示例内容列表
        logger.info("Creating sample content list...")
        content_list = create_sample_content_list()
        logger.info(f"Created content list with {len(content_list)} items")

        # 直接插入内容列表
        logger.info("\nInserting content list into RAGAnything...")
        await rag.insert_content_list(
            content_list=content_list,
            file_path="raganything_documentation.pdf",  # 用于引用的参考文件名
            split_by_character=None,  # 可选的文本分割
            split_by_character_only=False,  # 可选的文本分割模式
            doc_id="demo-doc-001",  # 自定义文档 ID
            display_stats=True,  # 显示内容统计
        )
        logger.info("Content list insertion completed!")

        # 查询示例 - 演示不同的查询方法
        logger.info("\nQuerying inserted content:")

        # 1. 使用 aquery() 的纯文本查询
        text_queries = [
            "What is RAGAnything and what are its main features?",
            "How does RAGAnything compare to traditional RAG systems?",
            "What are the technical specifications of the system?",
        ]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

        # 2. 使用 aquery_with_multimodal() 对特定多模态内容进行多模态查询
        logger.info(
            "\n[Multimodal Query]: Analyzing new performance data against existing benchmarks"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "Compare this new performance data with the existing benchmark results in the documentation",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Method,Accuracy,Speed,Memory
                                New_Approach,97.1%,110ms,1.9GB
                                Enhanced_RAG,91.4%,140ms,2.5GB""",
                    "table_caption": "Latest experimental results",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {multimodal_result}")

        # 3. 包含方程内容的另一个多模态查询
        logger.info("\n[Multimodal Query]: Mathematical formula analysis")
        equation_result = await rag.aquery_with_multimodal(
            "How does this similarity formula relate to the relevance scoring mentioned in the documentation?",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "sim(a, b) = \\frac{a \\cdot b}{||a|| \\times ||b||} + \\beta \\cdot context\\_weight",
                    "equation_caption": "Enhanced cosine similarity with context weighting",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {equation_result}")

        # 4. 使用不同的文档 ID 插入另一个内容列表
        logger.info("\nInserting additional content list...")
        additional_content = [
            {
                "type": "text",
                "text": "This is additional documentation about advanced features and configuration options.",
                "page_idx": 0,  # 此内容出现的页码
            },
            {
                "type": "table",
                "table_body": """| Configuration | Default Value | Range |
                                    |---------------|---------------|-------|
                                    | Chunk Size | 512 tokens | 128-2048 |
                                    | Context Window | 4096 tokens | 1024-8192 |
                                    | Batch Size | 32 | 1-128 |""",
                "table_caption": ["Advanced Configuration Parameters"],
                "page_idx": 1,  # 此表格出现的页码
            },
        ]

        await rag.insert_content_list(
            content_list=additional_content,
            file_path="advanced_configuration.pdf",
            doc_id="demo-doc-002",  # 不同的文档 ID
        )

        # 查询组合的知识库
        logger.info("\n[Combined Query]: What configuration options are available?")
        combined_result = await rag.aquery(
            "What configuration options are available and what are their default values?",
            mode="hybrid",
        )
        logger.info(f"Answer: {combined_result}")

    except Exception as e:
        logger.error(f"Error in content list insertion demo: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """运行示例的主函数"""
    parser = argparse.ArgumentParser(description="Insert Content List Example")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # 检查是否提供了 API 密钥
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # 运行演示
    asyncio.run(
        demo_insert_content_list(
            args.api_key,
            args.base_url,
            args.working_dir,
        )
    )


if __name__ == "__main__":
    # 首先配置日志记录
    configure_logging()

    print("RAGAnything Insert Content List Example")
    print("=" * 45)
    print("Demonstrating direct content list insertion without document parsing")
    print("=" * 45)

    main()
