"""
增强的 Markdown 到 PDF 转换

此模块提供改进的 Markdown 到 PDF 转换功能，包括：
- 更好的格式化和样式
- 图像支持
- 表格支持
- 代码语法高亮
- 自定义模板
- 多种输出格式
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import tempfile
import subprocess

try:
    import markdown

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    from weasyprint import HTML

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    # 检查 pandoc 模块是否存在（不直接使用，仅用于检测）
    import importlib.util

    spec = importlib.util.find_spec("pandoc")
    PANDOC_AVAILABLE = spec is not None
except ImportError:
    PANDOC_AVAILABLE = False


@dataclass
class MarkdownConfig:
    """Markdown 到 PDF 转换的配置"""

    # 样式选项
    css_file: Optional[str] = None
    template_file: Optional[str] = None
    page_size: str = "A4"
    margin: str = "1in"
    font_size: str = "12pt"
    line_height: str = "1.5"

    # 内容选项
    include_toc: bool = True
    syntax_highlighting: bool = True
    image_max_width: str = "100%"
    table_style: str = "border-collapse: collapse; width: 100%;"

    # 输出选项
    output_format: str = "pdf"  # pdf, html, docx
    output_dir: Optional[str] = None

    # 高级选项
    custom_css: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class EnhancedMarkdownConverter:
    """
    增强的 Markdown 到 PDF 转换器，支持多个后端

    支持多种转换方法：
    - WeasyPrint（推荐用于 HTML/CSS 样式）
    - Pandoc（推荐用于复杂文档）
    - ReportLab（备用，基本样式）
    """

    def __init__(self, config: Optional[MarkdownConfig] = None):
        """
        初始化转换器

        Args:
            config: 转换配置
        """
        self.config = config or MarkdownConfig()
        self.logger = logging.getLogger(__name__)

        # 检查可用的后端
        self.available_backends = self._check_backends()
        self.logger.info(f"Available backends: {list(self.available_backends.keys())}")

    def _check_backends(self) -> Dict[str, bool]:
        """检查哪些转换后端可用"""
        backends = {
            "weasyprint": WEASYPRINT_AVAILABLE,
            "pandoc": PANDOC_AVAILABLE,
            "markdown": MARKDOWN_AVAILABLE,
        }

        # 检查系统是否安装了 pandoc
        try:
            subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
            backends["pandoc_system"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            backends["pandoc_system"] = False

        return backends

    def _get_default_css(self) -> str:
        """获取默认 CSS 样式"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }

        h1 { font-size: 2em; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.2em; }
        h3 { font-size: 1.3em; }
        h4 { font-size: 1.1em; }

        p { margin-bottom: 1em; }

        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }

        pre code {
            background-color: transparent;
            padding: 0;
        }

        blockquote {
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #7f8c8d;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }

        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
        }

        ul, ol {
            margin-bottom: 1em;
        }

        li {
            margin-bottom: 0.5em;
        }

        a {
            color: #3498db;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        .toc {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 2em;
        }

        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }

        .toc li {
            margin-bottom: 0.3em;
        }

        .toc a {
            color: #2c3e50;
        }
        """

    def _process_markdown_content(self, content: str) -> str:
        """使用扩展处理 Markdown 内容"""
        if not MARKDOWN_AVAILABLE:
            raise RuntimeError(
                "Markdown library not available. Install with: pip install markdown"
            )

        # 配置 Markdown 扩展
        extensions = [
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.codehilite",
            "markdown.extensions.toc",
            "markdown.extensions.attr_list",
            "markdown.extensions.def_list",
            "markdown.extensions.footnotes",
        ]

        extension_configs = {
            "codehilite": {
                "css_class": "highlight",
                "use_pygments": True,
            },
            "toc": {
                "title": "Table of Contents",
                "permalink": True,
            },
        }

        # 将 Markdown 转换为 HTML
        md = markdown.Markdown(
            extensions=extensions, extension_configs=extension_configs
        )

        html_content = md.convert(content)

        # 添加 CSS 样式
        css = self.config.custom_css or self._get_default_css()

        # 创建完整的 HTML 文档
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Converted Document</title>
            <style>
                {css}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        return html_doc

    def convert_with_weasyprint(self, markdown_content: str, output_path: str) -> bool:
        """使用 WeasyPrint 进行转换（最适合样式化）"""
        if not WEASYPRINT_AVAILABLE:
            raise RuntimeError(
                "WeasyPrint not available. Install with: pip install weasyprint"
            )

        try:
            # 将 Markdown 处理为 HTML
            html_content = self._process_markdown_content(markdown_content)

            # 将 HTML 转换为 PDF
            html = HTML(string=html_content)
            html.write_pdf(output_path)

            self.logger.info(
                f"Successfully converted to PDF using WeasyPrint: {output_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"WeasyPrint conversion failed: {str(e)}")
            return False

    def convert_with_pandoc(
        self, markdown_content: str, output_path: str, use_system_pandoc: bool = False
    ) -> bool:
        """使用 Pandoc 进行转换（最适合复杂文档）"""
        if (
            not self.available_backends.get("pandoc_system", False)
            and not use_system_pandoc
        ):
            raise RuntimeError(
                "Pandoc not available. Install from: https://pandoc.org/installing.html"
            )

        temp_md_path = None
        try:
            import subprocess

            # 创建临时 markdown 文件
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            ) as temp_file:
                temp_file.write(markdown_content)
                temp_md_path = temp_file.name

            # 使用 wkhtmltopdf 引擎构建 pandoc 命令
            cmd = [
                "pandoc",
                temp_md_path,
                "-o",
                output_path,
                "--pdf-engine=wkhtmltopdf",
                "--standalone",
                "--toc",
                "--number-sections",
            ]

            # 运行 pandoc
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                self.logger.info(
                    f"Successfully converted to PDF using Pandoc: {output_path}"
                )
                return True
            else:
                self.logger.error(f"Pandoc conversion failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Pandoc conversion failed: {str(e)}")
            return False

        finally:
            if temp_md_path and os.path.exists(temp_md_path):
                try:
                    os.unlink(temp_md_path)
                except OSError as e:
                    self.logger.error(
                        f"Failed to clean up temp file {temp_md_path}: {str(e)}"
                    )

    def convert_markdown_to_pdf(
        self, markdown_content: str, output_path: str, method: str = "auto"
    ) -> bool:
        """
        将 markdown 内容转换为 PDF

        Args:
            markdown_content: 要转换的 Markdown 内容
            output_path: 输出 PDF 文件路径
            method: 转换方法（"auto"、"weasyprint"、"pandoc"、"pandoc_system"）

        Returns:
            如果转换成功则返回 True，否则返回 False
        """
        if method == "auto":
            method = self._get_recommended_backend()

        try:
            if method == "weasyprint":
                return self.convert_with_weasyprint(markdown_content, output_path)
            elif method == "pandoc":
                return self.convert_with_pandoc(markdown_content, output_path)
            elif method == "pandoc_system":
                return self.convert_with_pandoc(
                    markdown_content, output_path, use_system_pandoc=True
                )
            else:
                raise ValueError(f"Unknown conversion method: {method}")

        except Exception as e:
            self.logger.error(f"{method.title()} conversion failed: {str(e)}")
            return False

    def convert_file_to_pdf(
        self, input_path: str, output_path: Optional[str] = None, method: str = "auto"
    ) -> bool:
        """
        将 Markdown 文件转换为 PDF

        Args:
            input_path: 输入的 Markdown 文件路径
            output_path: 输出 PDF 文件路径（可选）
            method: 转换方法

        Returns:
            bool: 如果转换成功则返回 True
        """
        input_path_obj = Path(input_path)

        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # 读取 markdown 内容
        try:
            with open(input_path_obj, "r", encoding="utf-8") as f:
                markdown_content = f.read()
        except UnicodeDecodeError:
            # 尝试使用不同的编码
            for encoding in ["gbk", "latin-1", "cp1252"]:
                try:
                    with open(input_path_obj, "r", encoding=encoding) as f:
                        markdown_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise RuntimeError(
                    f"Could not decode file {input_path} with any supported encoding"
                )

        # 确定输出路径
        if output_path is None:
            output_path = str(input_path_obj.with_suffix(".pdf"))

        return self.convert_markdown_to_pdf(markdown_content, output_path, method)

    def get_backend_info(self) -> Dict[str, Any]:
        """获取可用后端的信息"""
        return {
            "available_backends": self.available_backends,
            "recommended_backend": self._get_recommended_backend(),
            "config": {
                "page_size": self.config.page_size,
                "margin": self.config.margin,
                "font_size": self.config.font_size,
                "include_toc": self.config.include_toc,
                "syntax_highlighting": self.config.syntax_highlighting,
            },
        }

    def _get_recommended_backend(self) -> str:
        """根据可用性获取推荐的后端"""
        if self.available_backends.get("pandoc_system", False):
            return "pandoc"
        elif self.available_backends.get("weasyprint", False):
            return "weasyprint"
        else:
            return "none"


def main():
    """增强 markdown 转换的命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Markdown to PDF conversion")
    parser.add_argument("input", nargs="?", help="Input markdown file")
    parser.add_argument("--output", "-o", help="Output PDF file")
    parser.add_argument(
        "--method",
        choices=["auto", "weasyprint", "pandoc", "pandoc_system"],
        default="auto",
        help="Conversion method",
    )
    parser.add_argument("--css", help="Custom CSS file")
    parser.add_argument("--info", action="store_true", help="Show backend information")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create converter
    config = MarkdownConfig()
    if args.css:
        config.css_file = args.css

    converter = EnhancedMarkdownConverter(config)

    # Show backend info if requested
    if args.info:
        info = converter.get_backend_info()
        print("Backend Information:")
        for backend, available in info["available_backends"].items():
            status = "✅" if available else "❌"
            print(f"  {status} {backend}")
        print(f"Recommended backend: {info['recommended_backend']}")
        return 0

    # Check if input file is provided
    if not args.input:
        parser.error("Input file is required when not using --info")

    # Convert file
    try:
        success = converter.convert_file_to_pdf(
            input_path=args.input, output_path=args.output, method=args.method
        )

        if success:
            print(f"✅ Successfully converted {args.input} to PDF")
            return 0
        else:
            print("❌ Conversion failed")
            return 1

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
