#!/usr/bin/env python3
"""
RAG-Anything Office 文档解析测试脚本

此脚本演示如何使用 MinerU 解析各种 Office 文档格式，
包括 DOC、DOCX、PPT、PPTX、XLS 和 XLSX 文件。

依赖项：
- 系统上已安装 LibreOffice
- RAG-Anything 包

用法：
    python office_document_test.py --file path/to/office/document.docx
"""

import argparse
import asyncio
import sys
from pathlib import Path
from raganything import RAGAnything


def check_libreoffice_installation():
    """检查 LibreOffice 是否已安装且可用"""
    import subprocess

    for cmd in ["libreoffice", "soffice"]:
        try:
            result = subprocess.run(
                [cmd, "--version"], capture_output=True, check=True, timeout=10
            )
            print(f"✅ LibreOffice found: {result.stdout.decode().strip()}")
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            continue

    print("❌ LibreOffice not found. Please install LibreOffice:")
    print("  - Windows: Download from https://www.libreoffice.org/download/download/")
    print("  - macOS: brew install --cask libreoffice")
    print("  - Ubuntu/Debian: sudo apt-get install libreoffice")
    print("  - CentOS/RHEL: sudo yum install libreoffice")
    return False


async def test_office_document_parsing(file_path: str):
    """使用 MinerU 测试 Office 文档解析"""

    print(f"🧪 Testing Office document parsing: {file_path}")

    # 检查文件是否存在且是否为支持的 Office 格式
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return False

    supported_extensions = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    if file_path.suffix.lower() not in supported_extensions:
        print(f"❌ Unsupported file format: {file_path.suffix}")
        print(f"   Supported formats: {', '.join(supported_extensions)}")
        return False

    print(f"📄 File format: {file_path.suffix.upper()}")
    print(f"📏 File size: {file_path.stat().st_size / 1024:.1f} KB")

    # 初始化 RAGAnything（仅用于解析功能）
    rag = RAGAnything()

    try:
        # 使用 MinerU 测试文档解析
        print("\n🔄 Testing document parsing with MinerU...")
        content_list, md_content = await rag.parse_document(
            file_path=str(file_path),
            output_dir="./test_output",
            parse_method="auto",
            display_stats=True,
        )

        print("✅ Parsing successful!")
        print(f"   📊 Content blocks: {len(content_list)}")
        print(f"   📝 Markdown length: {len(md_content)} characters")

        # 分析内容类型
        content_types = {}
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

        if content_types:
            print("   📋 Content distribution:")
            for content_type, count in sorted(content_types.items()):
                print(f"      • {content_type}: {count}")

        # 显示解析内容预览
        if md_content.strip():
            print("\n📄 Parsed content preview (first 500 characters):")
            preview = md_content.strip()[:500]
            print(f"   {preview}{'...' if len(md_content) > 500 else ''}")

        # 显示结构化内容示例
        text_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_items:
            print("\n📝 Sample text blocks:")
            for i, item in enumerate(text_items[:3], 1):
                text_content = item.get("text", "")
                if text_content.strip():
                    preview = text_content.strip()[:200]
                    print(
                        f"   {i}. {preview}{'...' if len(text_content) > 200 else ''}"
                    )

        # 检查图片
        image_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        if image_items:
            print(f"\n🖼️  Found {len(image_items)} image(s):")
            for i, item in enumerate(image_items, 1):
                print(f"   {i}. Image path: {item.get('img_path', 'N/A')}")

        # 检查表格
        table_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "table"
        ]
        if table_items:
            print(f"\n📊 Found {len(table_items)} table(s):")
            for i, item in enumerate(table_items, 1):
                table_body = item.get("table_body", "")
                row_count = len(table_body.split("\n"))
                print(f"   {i}. Table with {row_count} rows")

        print("\n🎉 Office document parsing test completed successfully!")
        print("📁 Output files saved to: ./test_output")
        return True

    except Exception as e:
        print(f"\n❌ Office document parsing failed: {str(e)}")
        import traceback

        print(f"   Full error: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Test Office document parsing with MinerU"
    )
    parser.add_argument("--file", help="Path to the Office document to test")
    parser.add_argument(
        "--check-libreoffice",
        action="store_true",
        help="Only check LibreOffice installation",
    )

    args = parser.parse_args()

    # 检查 LibreOffice 安装
    print("🔧 Checking LibreOffice installation...")
    if not check_libreoffice_installation():
        return 1

    if args.check_libreoffice:
        print("✅ LibreOffice installation check passed!")
        return 0

    # 如果不仅仅是检查依赖项，则需要 file 参数
    if not args.file:
        print(
            "❌ Error: --file argument is required when not using --check-libreoffice"
        )
        parser.print_help()
        return 1

    # 运行解析测试
    try:
        success = asyncio.run(test_office_document_parsing(args.file))
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
