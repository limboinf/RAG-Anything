"""
直接使用模态处理器的示例

此示例演示如何直接使用 RAG-Anything 的模态处理器而不通过 MinerU。
"""

import asyncio
import argparse
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import LightRAG
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
)

WORKING_DIR = "./rag_storage"


def get_llm_model_func(api_key: str, base_url: str = None):
    return (
        lambda prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    )


def get_vision_model_func(api_key: str, base_url: str = None):
    return (
        lambda prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        **kwargs: openai_complete_if_cache(
            "gpt-4o",
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
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
        if image_data
        else openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    )


async def process_image_example(lightrag: LightRAG, vision_model_func):
    """处理图像的示例"""
    # 创建图像处理器
    image_processor = ImageModalProcessor(
        lightrag=lightrag, modal_caption_func=vision_model_func
    )

    # 准备图像内容
    image_content = {
        "img_path": "image.jpg",
        "image_caption": ["Example image caption"],
        "image_footnote": ["Example image footnote"],
    }

    # 处理图像
    (description, entity_info, _) = await image_processor.process_multimodal_content(
        modal_content=image_content,
        content_type="image",
        file_path="image_example.jpg",
        entity_name="Example Image",
    )

    print("Image Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def process_table_example(lightrag: LightRAG, llm_model_func):
    """处理表格的示例"""
    # 创建表格处理器
    table_processor = TableModalProcessor(
        lightrag=lightrag, modal_caption_func=llm_model_func
    )

    # 准备表格内容
    table_content = {
        "table_body": """
        | Name | Age | Occupation |
        |------|-----|------------|
        | John | 25  | Engineer   |
        | Mary | 30  | Designer   |
        """,
        "table_caption": ["Employee Information Table"],
        "table_footnote": ["Data updated as of 2024"],
    }

    # 处理表格
    (description, entity_info, _) = await table_processor.process_multimodal_content(
        modal_content=table_content,
        content_type="table",
        file_path="table_example.md",
        entity_name="Employee Table",
    )

    print("\nTable Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def process_equation_example(lightrag: LightRAG, llm_model_func):
    """处理数学方程的示例"""
    # 创建方程处理器
    equation_processor = EquationModalProcessor(
        lightrag=lightrag, modal_caption_func=llm_model_func
    )

    # 准备方程内容
    equation_content = {"text": "E = mc^2", "text_format": "LaTeX"}

    # 处理方程
    (description, entity_info, _) = await equation_processor.process_multimodal_content(
        modal_content=equation_content,
        content_type="equation",
        file_path="equation_example.txt",
        entity_name="Mass-Energy Equivalence",
    )

    print("\nEquation Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def initialize_rag(api_key: str, base_url: str = None):
    # 使用环境变量进行嵌入配置
    import os

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
            ),
        ),
        llm_model_func=lambda prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    """运行示例的主函数"""
    parser = argparse.ArgumentParser(description="Modal Processors Example")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", help="Optional base URL for API")
    parser.add_argument(
        "--working-dir", "-w", default=WORKING_DIR, help="Working directory path"
    )

    args = parser.parse_args()

    # Run examples
    asyncio.run(main_async(args.api_key, args.base_url))


async def main_async(api_key: str, base_url: str = None):
    # 初始化 LightRAG
    lightrag = await initialize_rag(api_key, base_url)

    # 获取模型函数
    llm_model_func = get_llm_model_func(api_key, base_url)
    vision_model_func = get_vision_model_func(api_key, base_url)

    # 运行示例
    await process_image_example(lightrag, vision_model_func)
    await process_table_example(lightrag, llm_model_func)
    await process_equation_example(lightrag, llm_model_func)


if __name__ == "__main__":
    main()
