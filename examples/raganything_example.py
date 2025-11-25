#!/usr/bin/env python
"""
演示 MinerU 解析器与 RAGAnything 集成的示例脚本

此示例展示如何：
1. 使用 MinerU 解析器通过 RAGAnything 处理文档
2. 使用 aquery() 方法执行纯文本查询
3. 使用 aquery_with_multimodal() 方法执行包含特定多模态内容的查询
4. 在查询中处理不同类型的多模态内容（表格、公式）
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
    """为应用程序配置日志"""
    # 从环境变量获取日志目录路径，或使用当前目录
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # 从环境变量获取日志文件最大大小和备份数量
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

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


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    使用 RAGAnything 处理文档

    参数：
        file_path: 文档路径
        output_dir: RAG 结果的输出目录
        api_key: OpenAI API 密钥
        base_url: 可选的 API 基础 URL
        working_dir: RAG 存储的工作目录
    """
    try:
        # 创建 RAGAnything 配置
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # 解析器选择：mineru 或 docling
            parse_method="auto",  # 解析方法：auto、ocr 或 txt
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=True,
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
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # 如果提供了消息格式（用于多模态 VLM 增强查询），直接使用它
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # 传统的单图像格式
            elif image_data:
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
            # 纯文本格式
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

        # 使用新的数据类结构初始化 RAGAnything
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # 处理文档
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        # 示例查询 - 演示不同的查询方法
        logger.info("\n开始查询处理后的文档:")

        # 1. 使用 aquery() 进行纯文本查询
        text_queries = [
            "文档的主要内容是什么？",
            "文档中讨论了哪些关键主题？",
        ]

        for query in text_queries:
            logger.info(f"\n[文本查询]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"回答: {result}")

        # 2. 使用 aquery_with_multimodal() 进行包含特定多模态内容的查询
        # logger.info(
        #     "\n[多模态查询]: 在文档上下文中分析性能数据"
        # )
        # multimodal_result = await rag.aquery_with_multimodal(
        #     "将此性能数据与文档中提到的任何类似结果进行比较",
        #     multimodal_content=[
        #         {
        #             "type": "table",
        #             "table_data": """Method,Accuracy,Processing_Time
        #                         RAGAnything,95.2%,120ms
        #                         Traditional_RAG,87.3%,180ms
        #                         Baseline,82.1%,200ms""",
        #             "table_caption": "Performance comparison results",
        #         }
        #     ],
        #     mode="hybrid",
        # )
        # logger.info(f"多模态查询回答: {multimodal_result}")

        # # 3. 另一个包含公式内容的多模态查询
        # logger.info("\n[多模态查询]: 数学公式分析")
        # equation_result = await rag.aquery_with_multimodal(
        #     "解释这个公式并将其与文档中的任何数学概念联系起来",
        #     multimodal_content=[
        #         {
        #             "type": "equation",
        #             "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
        #             "equation_caption": "F1-score calculation formula",
        #         }
        #     ],
        #     mode="hybrid",
        # )
        # logger.info(f"公式分析回答: {equation_result}")

    except Exception as e:
        logger.error(f"处理 RAG 时出错: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """运行示例的主函数"""
    parser = argparse.ArgumentParser(description="MinerU RAG 示例")
    parser.add_argument("file_path", help="需要处理的文档路径")
    parser.add_argument(
        "--working_dir",
        "-w",
        default="./rag_storage",
        help="RAG 存储工作目录（默认 ./rag_storage）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="结果输出目录（默认 ./output）",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API Key，默认读取 LLM_BINDING_API_KEY 环境变量",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="可选的 API 基础 URL（默认读取 LLM_BINDING_HOST）",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="解析器名称（默认使用 PARSER 环境变量，缺省为 mineru）",
    )

    args = parser.parse_args()

    # 检查是否提供了 API 密钥
    if not args.api_key:
        logger.error("错误：需要提供 OpenAI API Key")
        logger.error("请设置对应环境变量或通过 --api-key 选项传入")
        return

    # 如果指定，创建输出目录
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # 使用 RAG 处理
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # 首先配置日志
    configure_logging()

    print("RAGAnything 示例")
    print("=" * 30)
    print("使用多模态 RAG 流水线处理文档")
    print("=" * 30)

    main()
