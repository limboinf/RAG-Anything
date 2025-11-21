"""
LM Studio 与 RAG-Anything 集成示例

此示例演示如何将 LM Studio 与 RAG-Anything 集成以进行本地文本文档处理和查询。

要求：
- 本地运行的 LM Studio 并启用服务器
- OpenAI Python 包：pip install openai
- 已安装 RAG-Anything：pip install raganything

环境设置：
创建包含以下内容的 .env 文件：
LLM_BINDING=lmstudio
LLM_MODEL=openai/gpt-oss-20b
LLM_BINDING_HOST=http://localhost:1234/v1
LLM_BINDING_API_KEY=lm-studio
EMBEDDING_BINDING=lmstudio
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
EMBEDDING_BINDING_HOST=http://localhost:1234/v1
EMBEDDING_BINDING_API_KEY=lm-studio
"""

import os
import uuid
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 加载环境变量
load_dotenv()

# RAG-Anything 导入
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

LM_BASE_URL = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
LM_MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LM_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")


async def lmstudio_llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """LightRAG 的顶层 LLM 函数（可 pickle 序列化）。"""
    return await openai_complete_if_cache(
        model=LM_MODEL_NAME,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=LM_BASE_URL,
        api_key=LM_API_KEY,
        **kwargs,
    )


async def lmstudio_embedding_async(texts: List[str]) -> List[List[float]]:
    """LightRAG 的顶层嵌入函数（可 pickle 序列化）。"""
    from lightrag.llm.openai import openai_embed

    embeddings = await openai_embed(
        texts=texts,
        model=LM_EMBED_MODEL,
        base_url=LM_BASE_URL,
        api_key=LM_API_KEY,
    )
    return embeddings.tolist()


class LMStudioRAGIntegration:
    """LM Studio 与 RAG-Anything 的集成类。"""

    def __init__(self):
        # 使用标准 LLM_BINDING 变量的 LM Studio 配置
        self.base_url = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
        self.api_key = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
        self.model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
        )

        # RAG-Anything 配置
        # 每次运行使用新的工作目录以避免旧版 doc_status 模式冲突
        self.config = RAGAnythingConfig(
            working_dir=f"./rag_storage_lmstudio/{uuid.uuid4()}",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        print(f"📁 Using working_dir: {self.config.working_dir}")

        self.rag = None

    async def test_connection(self) -> bool:
        """测试 LM Studio 连接。"""
        try:
            print(f"🔌 Testing LM Studio connection at: {self.base_url}")
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            models = await client.models.list()
            print(f"✅ Connected successfully! Found {len(models.data)} models")

            # 显示可用模型
            print("📊 Available models:")
            for i, model in enumerate(models.data[:5]):
                marker = "🎯" if model.id == self.model_name else "  "
                print(f"{marker} {i+1}. {model.id}")

            if len(models.data) > 5:
                print(f"  ... and {len(models.data) - 5} more models")

            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("1. Ensure LM Studio is running")
            print("2. Start the local server in LM Studio")
            print("3. Load a model or enable just-in-time loading")
            print(f"4. Verify server address: {self.base_url}")
            return False
        finally:
            try:
                await client.close()
            except Exception:
                pass

    async def test_chat_completion(self) -> bool:
        """测试基本聊天功能。"""
        try:
            print(f"💬 Testing chat with model: {self.model_name}")
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {
                        "role": "user",
                        "content": "Hello! Please confirm you're working and tell me your capabilities.",
                    },
                ],
                max_tokens=100,
                temperature=0.7,
            )

            result = response.choices[0].message.content.strip()
            print("✅ Chat test successful!")
            print(f"Response: {result}")
            return True
        except Exception as e:
            print(f"❌ Chat test failed: {str(e)}")
            return False
        finally:
            try:
                await client.close()
            except Exception:
                pass

    # 已移除已弃用的工厂辅助函数以减少冗余

    def embedding_func_factory(self):
        """创建完全可序列化的嵌入函数。"""
        return EmbeddingFunc(
            embedding_dim=768,  # nomic-embed-text-v1.5 默认维度
            max_token_size=8192,  # nomic-embed-text-v1.5 上下文长度
            func=lmstudio_embedding_async,
        )

    async def initialize_rag(self):
        """使用 LM Studio 函数初始化 RAG-Anything。"""
        print("Initializing RAG-Anything with LM Studio...")

        try:
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=lmstudio_llm_model_func,
                embedding_func=self.embedding_func_factory(),
            )

            # 兼容性：避免将未知字段 'multimodal_processed' 写入 LightRAG doc_status
            # 较旧的 LightRAG 版本可能不接受 DocProcessingStatus 中的此额外字段
            async def _noop_mark_multimodal(doc_id: str):
                return None

            self.rag._mark_multimodal_processing_complete = _noop_mark_multimodal

            print("✅ RAG-Anything initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ RAG initialization failed: {str(e)}")
            return False

    async def process_document_example(self, file_path: str):
        """示例：使用 LM Studio 后端处理文档。"""
        if not self.rag:
            print("❌ RAG not initialized. Call initialize_rag() first.")
            return

        try:
            print(f"📄 Processing document: {file_path}")
            await self.rag.process_document_complete(
                file_path=file_path,
                output_dir="./output_lmstudio",
                parse_method="auto",
                display_stats=True,
            )
            print("✅ Document processing completed!")
        except Exception as e:
            print(f"❌ Document processing failed: {str(e)}")

    async def query_examples(self):
        """使用不同模式的查询示例。"""
        if not self.rag:
            print("❌ RAG not initialized. Call initialize_rag() first.")
            return

        # 查询示例
        queries = [
            ("What are the main topics in the processed documents?", "hybrid"),
            ("Summarize any tables or data found in the documents", "local"),
            ("What images or figures are mentioned?", "global"),
        ]

        print("\n🔍 Running example queries...")
        for query, mode in queries:
            try:
                print(f"\nQuery ({mode}): {query}")
                result = await self.rag.aquery(query, mode=mode)
                print(f"Answer: {result[:200]}...")
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")

    async def simple_query_example(self):
        """使用示例内容的基本文本查询示例。"""
        if not self.rag:
            print("❌ RAG not initialized")
            return

        try:
            print("\nAdding sample content for testing...")

            # 创建 RAGAnything 期望格式的内容列表
            content_list = [
                {
                    "type": "text",
                    "text": """LM Studio Integration with RAG-Anything

This integration demonstrates how to connect LM Studio's local AI models with RAG-Anything's document processing capabilities. The system uses:

- LM Studio for local LLM inference
- nomic-embed-text-v1.5 for embeddings (768 dimensions)
- RAG-Anything for document processing and retrieval

Key benefits include:
- Privacy: All processing happens locally
- Performance: Direct API access to local models
- Flexibility: Support for various document formats
- Cost-effective: No external API usage""",
                    "page_idx": 0,
                }
            ]

            # 使用正确的方法插入内容列表
            await self.rag.insert_content_list(
                content_list=content_list,
                file_path="lmstudio_integration_demo.txt",
                # 使用唯一的 doc_id 以避免冲突和跨运行重用 doc_status
                doc_id=f"demo-content-{uuid.uuid4()}",
                display_stats=True,
            )
            print("✅ Sample content added to knowledge base")

            print("\nTesting basic text query...")

            # 简单文本查询示例
            result = await self.rag.aquery(
                "What are the key benefits of this LM Studio integration?",
                mode="hybrid",
            )
            print(f"✅ Query result: {result[:300]}...")

        except Exception as e:
            print(f"❌ Query failed: {str(e)}")


async def main():
    """主示例函数。"""
    print("=" * 70)
    print("LM Studio + RAG-Anything Integration Example")
    print("=" * 70)

    # 初始化集成
    integration = LMStudioRAGIntegration()

    # 测试连接
    if not await integration.test_connection():
        return False

    print()
    if not await integration.test_chat_completion():
        return False

    # 初始化 RAG
    print("\n" + "─" * 50)
    if not await integration.initialize_rag():
        return False

    # 文档处理示例（取消注释并提供真实文件路径）
    # await integration.process_document_example("path/to/your/document.pdf")

    # 查询示例（处理文档后取消注释）
    # await integration.query_examples()

    # 基本查询示例
    await integration.simple_query_example()

    print("\n" + "=" * 70)
    print("Integration example completed successfully!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    print("🚀 Starting LM Studio integration example...")
    success = asyncio.run(main())

    exit(0 if success else 1)
