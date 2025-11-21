# type: ignore
"""
通用文档解析工具

本模块提供使用 MinerU 2.0 库解析 PDF 和图像文档的功能,
并将解析结果转换为 markdown 和 JSON 格式

注意: MinerU 2.0 不再包含 LibreOffice 文档转换模块。
对于 Office 文档 (.doc, .docx, .ppt, .pptx),请先将它们转换为 PDF 格式。
"""

from __future__ import annotations


import json
import argparse
import base64
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Any,
    TypeVar,
)

T = TypeVar("T")


class MineruExecutionError(Exception):
    """捕获 mineru 错误"""

    def __init__(self, return_code, error_msg):
        self.return_code = return_code
        self.error_msg = error_msg
        super().__init__(
            f"Mineru command failed with return code {return_code}: {error_msg}"
        )


class Parser:
    """
    文档解析工具的基类。

    定义用于解析不同文档类型的通用功能和常量。
    """

    # 定义常见文件格式
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}

    # 类级别的日志记录器
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """初始化基础解析器。"""
        pass

    @staticmethod
    def convert_office_to_pdf(
        doc_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        将 Office 文档 (.doc, .docx, .ppt, .pptx, .xls, .xlsx) 转换为 PDF。
        需要安装 LibreOffice。

        Args:
            doc_path: Office 文档文件的路径
            output_dir: PDF 文件的输出目录

        Returns:
            生成的 PDF 文件的路径
        """
        try:
            # 转换为 Path 对象以便于处理
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Office document does not exist: {doc_path}")

            name_without_suff = doc_path.stem

            # 准备输出目录
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "libreoffice_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # 为 PDF 转换创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # 使用 LibreOffice 转换为 PDF
                logging.info(f"Converting {doc_path.name} to PDF using LibreOffice...")

                # 准备子进程参数以在 Windows 上隐藏控制台窗口
                import platform

                # 按优先顺序尝试 LibreOffice 命令
                commands_to_try = ["libreoffice", "soffice"]

                conversion_successful = False
                for cmd in commands_to_try:
                    try:
                        convert_cmd = [
                            cmd,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(temp_path),
                            str(doc_path),
                        ]

                        # 准备转换子进程参数
                        convert_subprocess_kwargs = {
                            "capture_output": True,
                            "text": True,
                            "timeout": 60,  # 60 秒超时
                            "encoding": "utf-8",
                            "errors": "ignore",
                        }

                        # 在 Windows 上隐藏控制台窗口
                        if platform.system() == "Windows":
                            convert_subprocess_kwargs["creationflags"] = (
                                subprocess.CREATE_NO_WINDOW
                            )

                        result = subprocess.run(
                            convert_cmd, **convert_subprocess_kwargs
                        )

                        if result.returncode == 0:
                            conversion_successful = True
                            logging.info(
                                f"Successfully converted {doc_path.name} to PDF using {cmd}"
                            )
                            break
                        else:
                            logging.warning(
                                f"LibreOffice command '{cmd}' failed: {result.stderr}"
                            )
                    except FileNotFoundError:
                        logging.warning(f"LibreOffice command '{cmd}' not found")
                    except subprocess.TimeoutExpired:
                        logging.warning(f"LibreOffice command '{cmd}' timed out")
                    except Exception as e:
                        logging.error(
                            f"LibreOffice command '{cmd}' failed with exception: {e}"
                        )

                if not conversion_successful:
                    raise RuntimeError(
                        f"LibreOffice conversion failed for {doc_path.name}. "
                        f"Please ensure LibreOffice is installed:\n"
                        "- Windows: Download from https://www.libreoffice.org/download/download/\n"
                        "- macOS: brew install --cask libreoffice\n"
                        "- Ubuntu/Debian: sudo apt-get install libreoffice\n"
                        "- CentOS/RHEL: sudo yum install libreoffice\n"
                        "Alternatively, convert the document to PDF manually."
                    )

                # 查找生成的 PDF
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    raise RuntimeError(
                        f"PDF conversion failed for {doc_path.name} - no PDF file generated. "
                        f"Please check LibreOffice installation or try manual conversion."
                    )

                pdf_path = pdf_files[0]
                logging.info(
                    f"Generated PDF: {pdf_path.name} ({pdf_path.stat().st_size} bytes)"
                )

                # 验证生成的 PDF
                if pdf_path.stat().st_size < 100:  # 非常小的文件,可能是空的
                    raise RuntimeError(
                        "Generated PDF appears to be empty or corrupted. "
                        "Original file may have issues or LibreOffice conversion failed."
                    )

                # 将 PDF 复制到最终输出目录
                final_pdf_path = base_output_dir / f"{name_without_suff}.pdf"
                import shutil

                shutil.copy2(pdf_path, final_pdf_path)

                return final_pdf_path

        except Exception as e:
            logging.error(f"Error in convert_office_to_pdf: {str(e)}")
            raise

    @staticmethod
    def convert_text_to_pdf(
        text_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        使用 ReportLab 将文本文件 (.txt, .md) 转换为 PDF,支持完整的 markdown。

        Args:
            text_path: 文本文件的路径
            output_dir: PDF 文件的输出目录

        Returns:
            生成的 PDF 文件的路径
        """
        try:
            text_path = Path(text_path)
            if not text_path.exists():
                raise FileNotFoundError(f"Text file does not exist: {text_path}")

            # 支持的文本格式
            supported_text_formats = {".txt", ".md"}
            if text_path.suffix.lower() not in supported_text_formats:
                raise ValueError(f"Unsupported text format: {text_path.suffix}")

            # 读取文本内容
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                # 尝试使用不同的编码
                for encoding in ["gbk", "latin-1", "cp1252"]:
                    try:
                        with open(text_path, "r", encoding=encoding) as f:
                            text_content = f.read()
                        logging.info(f"Successfully read file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(
                        f"Could not decode text file {text_path.name} with any supported encoding"
                    )

            # 准备输出目录
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = text_path.parent / "reportlab_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = base_output_dir / f"{text_path.stem}.pdf"

            # 将文本转换为 PDF
            logging.info(f"Converting {text_path.name} to PDF...")

            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont

                support_chinese = True
                try:
                    if "WenQuanYi" not in pdfmetrics.getRegisteredFontNames():
                        if not Path(
                            "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
                        ).exists():
                            support_chinese = False
                            logging.warning(
                                "WenQuanYi font not found at /usr/share/fonts/wqy-microhei/wqy-microhei.ttc. Chinese characters may not render correctly."
                            )
                        else:
                            pdfmetrics.registerFont(
                                TTFont(
                                    "WenQuanYi",
                                    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
                                )
                            )
                except Exception as e:
                    support_chinese = False
                    logging.warning(
                        f"Failed to register WenQuanYi font: {e}. Chinese characters may not render correctly."
                    )

                # 创建 PDF 文档
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=A4,
                    leftMargin=inch,
                    rightMargin=inch,
                    topMargin=inch,
                    bottomMargin=inch,
                )

                # 获取样式
                styles = getSampleStyleSheet()
                normal_style = styles["Normal"]
                heading_style = styles["Heading1"]
                if support_chinese:
                    normal_style.fontName = "WenQuanYi"
                    heading_style.fontName = "WenQuanYi"

                # 尝试注册支持中文字符的字体
                try:
                    # 尝试使用支持中文的系统字体
                    import platform

                    system = platform.system()
                    if system == "Windows":
                        # 尝试常见的 Windows 字体
                        for font_name in ["SimSun", "SimHei", "Microsoft YaHei"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                    elif system == "Darwin":  # macOS
                        for font_name in ["STSong-Light", "STHeiti"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                except Exception:
                    pass  # 如果中文字体设置失败,使用默认字体

                # 构建内容
                story = []

                # 处理 markdown 或纯文本
                if text_path.suffix.lower() == ".md":
                    # 处理 markdown 内容 - 简化实现
                    lines = text_content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            story.append(Spacer(1, 12))
                            continue

                        # 标题
                        if line.startswith("#"):
                            level = len(line) - len(line.lstrip("#"))
                            header_text = line.lstrip("#").strip()
                            if header_text:
                                header_style = ParagraphStyle(
                                    name=f"Heading{level}",
                                    parent=heading_style,
                                    fontSize=max(16 - level, 10),
                                    spaceAfter=8,
                                    spaceBefore=16 if level <= 2 else 12,
                                )
                                story.append(Paragraph(header_text, header_style))
                        else:
                            # 常规文本
                            story.append(Paragraph(line, normal_style))
                            story.append(Spacer(1, 6))
                else:
                    # 处理纯文本文件 (.txt)
                    logging.info(
                        f"Processing plain text file with {len(text_content)} characters..."
                    )

                    # 将文本分割为行并处理每一行
                    lines = text_content.split("\n")
                    line_count = 0

                    for line in lines:
                        line = line.rstrip()
                        line_count += 1

                        # 空行
                        if not line.strip():
                            story.append(Spacer(1, 6))
                            continue

                        # 常规文本行
                        # 转义 ReportLab 的特殊字符
                        safe_line = (
                            line.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )

                        # 创建段落
                        story.append(Paragraph(safe_line, normal_style))
                        story.append(Spacer(1, 3))

                    logging.info(f"Added {line_count} lines to PDF")

                    # 如果没有添加任何内容,添加占位符
                    if not story:
                        story.append(Paragraph("(Empty text file)", normal_style))

                # 构建 PDF
                doc.build(story)
                logging.info(
                    f"Successfully converted {text_path.name} to PDF ({pdf_path.stat().st_size / 1024:.1f} KB)"
                )

            except ImportError:
                raise RuntimeError(
                    "reportlab is required for text-to-PDF conversion. "
                    "Please install it using: pip install reportlab"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert text file {text_path.name} to PDF: {str(e)}"
                )

            # 验证生成的 PDF
            if not pdf_path.exists() or pdf_path.stat().st_size < 100:
                raise RuntimeError(
                    f"PDF conversion failed for {text_path.name} - generated PDF is empty or corrupted."
                )

            return pdf_path

        except Exception as e:
            logging.error(f"Error in convert_text_to_pdf: {str(e)}")
            raise

    @staticmethod
    def _process_inline_markdown(text: str) -> str:
        """
        处理内联 markdown 格式化 (粗体、斜体、代码、链接)

        Args:
            text: 带有 markdown 格式的原始文本

        Returns:
            带有 ReportLab 标记的文本
        """
        import re

        # 转义 ReportLab 的特殊字符
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 粗体文本: **text** 或 __text__
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

        # 斜体文本: *text* 或 _text_ (但不在单词中间)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

        # 内联代码: `code`
        text = re.sub(
            r"`([^`]+?)`",
            r'<font name="Courier" size="9" color="darkred">\1</font>',
            text,
        )

        # 链接: [text](url) - 转换为带有 URL 注释的文本
        def link_replacer(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<link href="{url}" color="blue"><u>{link_text}</u></link>'

        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", link_replacer, text)

        # 删除线: ~~text~~
        text = re.sub(r"~~(.*?)~~", r"<strike>\1</strike>", text)

        return text

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        解析 PDF 文档的抽象方法。
        必须由子类实现。

        Args:
            pdf_path: PDF 文件的路径
            output_dir: 输出目录路径
            method: 解析方法 (auto, txt, ocr)
            lang: 用于 OCR 优化的文档语言
            **kwargs: 解析器特定命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        raise NotImplementedError("parse_pdf must be implemented by subclasses")

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        解析图像文档的抽象方法。
        必须由子类实现。

        注意: 不同的解析器可能支持不同的图像格式。
        请查看特定解析器的文档以了解支持的格式。

        Args:
            image_path: 图像文件的路径
            output_dir: 输出目录路径
            lang: 用于 OCR 优化的文档语言
            **kwargs: 解析器特定命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        raise NotImplementedError("parse_image must be implemented by subclasses")

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        解析文档的抽象方法。
        必须由子类实现。

        Args:
            file_path: 要解析的文件路径
            method: 解析方法 (auto, txt, ocr)
            output_dir: 输出目录路径
            lang: 用于 OCR 优化的文档语言
            **kwargs: 解析器特定命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        raise NotImplementedError("parse_document must be implemented by subclasses")

    def check_installation(self) -> bool:
        """
        检查解析器是否正确安装的抽象方法。
        必须由子类实现。

        Returns:
            bool: 如果安装有效则为 True,否则为 False
        """
        raise NotImplementedError(
            "check_installation must be implemented by subclasses"
        )


class MineruParser(Parser):
    """
    MinerU 2.0 文档解析工具类

    支持解析 PDF 和图像文档,将内容转换为结构化数据
    并生成 markdown 和 JSON 输出。

    注意: 不再直接支持 Office 文档。请先将它们转换为 PDF。
    """

    __slots__ = ()

    # 类级别的日志记录器
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """初始化 MineruParser"""
        super().__init__()

    @staticmethod
    def _run_mineru_command(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        method: str = "auto",
        lang: Optional[str] = None,
        backend: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        formula: bool = True,
        table: bool = True,
        device: Optional[str] = None,
        source: Optional[str] = None,
        vlm_url: Optional[str] = None,
    ) -> None:
        """
        运行 mineru 命令行工具

        Args:
            input_path: 输入文件或目录的路径
            output_dir: 输出目录路径
            method: 解析方法 (auto, txt, ocr)
            lang: 用于 OCR 优化的文档语言
            backend: 解析后端
            start_page: 起始页码 (从0开始)
            end_page: 结束页码 (从0开始)
            formula: 启用公式解析
            table: 启用表格解析
            device: 推理设备
            source: 模型源
            vlm_url: 当后端为 `vlm-sglang-client` 时,需要指定 server_url
        """
        cmd = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-m",
            method,
        ]

        if backend:
            cmd.extend(["-b", backend])
        if source:
            cmd.extend(["--source", source])
        if lang:
            cmd.extend(["-l", lang])
        if start_page is not None:
            cmd.extend(["-s", str(start_page)])
        if end_page is not None:
            cmd.extend(["-e", str(end_page)])
        if not formula:
            cmd.extend(["-f", "false"])
        if not table:
            cmd.extend(["-t", "false"])
        if device:
            cmd.extend(["-d", device])
        if vlm_url:
            cmd.extend(["-u", vlm_url])

        output_lines = []
        error_lines = []

        try:
            # 准备子进程参数以在 Windows 上隐藏控制台窗口
            import platform
            import threading
            from queue import Queue, Empty

            # 记录正在执行的命令
            logging.info(f"Executing mineru command: {' '.join(cmd)}")

            subprocess_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "ignore",
                "bufsize": 1,  # 行缓冲
            }

            # 在 Windows 上隐藏控制台窗口
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            # 从子进程读取输出并添加到队列的函数
            def enqueue_output(pipe, queue, prefix):
                try:
                    for line in iter(pipe.readline, ""):
                        if line.strip():  # 只添加非空行
                            queue.put((prefix, line.strip()))
                    pipe.close()
                except Exception as e:
                    queue.put((prefix, f"Error reading {prefix}: {e}"))

            # 启动子进程
            process = subprocess.Popen(cmd, **subprocess_kwargs)

            # 为 stdout 和 stderr 创建队列
            stdout_queue = Queue()
            stderr_queue = Queue()

            # 启动线程读取输出
            stdout_thread = threading.Thread(
                target=enqueue_output, args=(process.stdout, stdout_queue, "STDOUT")
            )
            stderr_thread = threading.Thread(
                target=enqueue_output, args=(process.stderr, stderr_queue, "STDERR")
            )

            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # 实时处理输出
            while process.poll() is None:
                # 检查 stdout 队列
                try:
                    while True:
                        prefix, line = stdout_queue.get_nowait()
                        output_lines.append(line)
                        # 以 INFO 级别记录 mineru 输出,前缀为 [MinerU]
                        logging.info(f"[MinerU] {line}")
                except Empty:
                    pass

                # 检查 stderr 队列
                try:
                    while True:
                        prefix, line = stderr_queue.get_nowait()
                        # 以 WARNING 级别记录 mineru 错误
                        if "warning" in line.lower():
                            logging.warning(f"[MinerU] {line}")
                        elif "error" in line.lower():
                            logging.error(f"[MinerU] {line}")
                            error_message = line.split("\n")[0]
                            error_lines.append(error_message)
                        else:
                            logging.info(f"[MinerU] {line}")
                except Empty:
                    pass

                # 小延迟以防止忙等待
                import time

                time.sleep(0.1)

            # 处理进程完成后的任何剩余输出
            try:
                while True:
                    prefix, line = stdout_queue.get_nowait()
                    output_lines.append(line)
                    logging.info(f"[MinerU] {line}")
            except Empty:
                pass

            try:
                while True:
                    prefix, line = stderr_queue.get_nowait()
                    if "warning" in line.lower():
                        logging.warning(f"[MinerU] {line}")
                    elif "error" in line.lower():
                        logging.error(f"[MinerU] {line}")
                        error_message = line.split("\n")[0]
                        error_lines.append(error_message)
                    else:
                        logging.info(f"[MinerU] {line}")
            except Empty:
                pass

            # 等待进程完成并获取返回码
            return_code = process.wait()

            # 等待线程完成
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            if return_code != 0 or error_lines:
                logging.info("[MinerU] Command executed failed")
                raise MineruExecutionError(return_code, error_lines)
            else:
                logging.info("[MinerU] Command executed successfully")

        except MineruExecutionError:
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running mineru subprocess command: {e}")
            logging.error(f"Command: {' '.join(cmd)}")
            logging.error(f"Return code: {e.returncode}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "mineru command not found. Please ensure MinerU 2.0 is properly installed:\n"
                "pip install -U 'mineru[core]' or uv pip install -U 'mineru[core]'"
            )
        except Exception as e:
            error_message = f"Unexpected error running mineru command: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message) from e

    @staticmethod
    def _read_output_files(
        output_dir: Path, file_stem: str, method: str = "auto"
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        读取由 mineru 生成的输出文件

        Args:
            output_dir: 输出目录
            file_stem: 不带扩展名的文件名

        Returns:
            包含 (内容列表 JSON, Markdown 文本) 的元组
        """
        # 查找生成的文件
        md_file = output_dir / f"{file_stem}.md"
        json_file = output_dir / f"{file_stem}_content_list.json"
        images_base_dir = output_dir  # 图像的基础目录

        file_stem_subdir = output_dir / file_stem
        if file_stem_subdir.exists():
            md_file = file_stem_subdir / method / f"{file_stem}.md"
            json_file = file_stem_subdir / method / f"{file_stem}_content_list.json"
            images_base_dir = file_stem_subdir / method

        # 读取 markdown 内容
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logging.warning(f"Could not read markdown file {md_file}: {e}")

        # 读取 JSON 内容列表
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content_list = json.load(f)

                # 始终将 content_list 中的相对路径修正为绝对路径
                logging.info(
                    f"Fixing image paths in {json_file} with base directory: {images_base_dir}"
                )
                for item in content_list:
                    if isinstance(item, dict):
                        for field_name in [
                            "img_path",
                            "table_img_path",
                            "equation_img_path",
                        ]:
                            if field_name in item and item[field_name]:
                                img_path = item[field_name]
                                absolute_img_path = (
                                    images_base_dir / img_path
                                ).resolve()
                                item[field_name] = str(absolute_img_path)
                                logging.debug(
                                    f"Updated {field_name}: {img_path} -> {item[field_name]}"
                                )

            except Exception as e:
                logging.warning(f"Could not read JSON file {json_file}: {e}")

        return content_list, md_content

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        使用 MinerU 2.0 解析 PDF 文档

        Args:
            pdf_path: PDF 文件的路径
            output_dir: 输出目录路径
            method: 解析方法 (auto, txt, ocr)
            lang: 用于 OCR 优化的文档语言
            **kwargs: mineru 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        try:
            # 转换为 Path 对象以便于处理
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # 准备输出目录
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "mineru_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # 运行 mineru 命令
            self._run_mineru_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                method=method,
                lang=lang,
                **kwargs,
            )

            # 读取生成的输出文件
            backend = kwargs.get("backend", "")
            if backend.startswith("vlm-"):
                method = "vlm"

            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff, method=method
            )
            return content_list

        except MineruExecutionError:
            raise
        except Exception as e:
            logging.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        使用 MinerU 2.0 解析图像文档

        注意: MinerU 2.0 原生支持 .png, .jpeg, .jpg 格式。
        其他格式 (.bmp, .tiff, .tif 等) 将自动转换为 .png。

        Args:
            image_path: 图像文件的路径
            output_dir: 输出目录路径
            lang: 用于 OCR 优化的文档语言
            **kwargs: mineru 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        try:
            # 转换为 Path 对象以便于处理
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")

            # MinerU 2.0 支持的图像格式
            mineru_supported_formats = {".png", ".jpeg", ".jpg"}

            # 所有支持的图像格式 (包括我们可以转换的格式)
            all_supported_formats = {
                ".png",
                ".jpeg",
                ".jpg",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            }

            ext = image_path.suffix.lower()
            if ext not in all_supported_formats:
                raise ValueError(
                    f"Unsupported image format: {ext}. Supported formats: {', '.join(all_supported_formats)}"
                )

            # 确定要处理的实际图像文件
            actual_image_path = image_path
            temp_converted_file = None

            # 如果格式不是 MinerU 原生支持的,则进行转换
            if ext not in mineru_supported_formats:
                logging.info(
                    f"Converting {ext} image to PNG for MinerU compatibility..."
                )

                try:
                    from PIL import Image
                except ImportError:
                    raise RuntimeError(
                        "PIL/Pillow is required for image format conversion. "
                        "Please install it using: pip install Pillow"
                    )

                # 为转换创建临时目录
                temp_dir = Path(tempfile.mkdtemp())
                temp_converted_file = temp_dir / f"{image_path.stem}_converted.png"

                try:
                    # 打开并转换图像
                    with Image.open(image_path) as img:
                        # 处理不同的图像模式
                        if img.mode in ("RGBA", "LA", "P"):
                            # 对于带有透明度或调色板的图像,首先转换为 RGB
                            if img.mode == "P":
                                img = img.convert("RGBA")

                            # 为透明图像创建白色背景
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            if img.mode == "RGBA":
                                background.paste(
                                    img, mask=img.split()[-1]
                                )  # 使用 alpha 通道作为遮罩
                            else:
                                background.paste(img)
                            img = background
                        elif img.mode not in ("RGB", "L"):
                            # 将其他模式转换为 RGB
                            img = img.convert("RGB")

                        # 保存为 PNG
                        img.save(temp_converted_file, "PNG", optimize=True)
                        logging.info(
                            f"Successfully converted {image_path.name} to PNG ({temp_converted_file.stat().st_size / 1024:.1f} KB)"
                        )

                        actual_image_path = temp_converted_file

                except Exception as e:
                    if temp_converted_file and temp_converted_file.exists():
                        temp_converted_file.unlink()
                    raise RuntimeError(
                        f"Failed to convert image {image_path.name}: {str(e)}"
                    )

            name_without_suff = image_path.stem

            # 准备输出目录
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = image_path.parent / "mineru_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 运行 mineru 命令 (图像使用 OCR 方法处理)
                self._run_mineru_command(
                    input_path=actual_image_path,
                    output_dir=base_output_dir,
                    method="ocr",  # 图像需要 OCR 方法
                    lang=lang,
                    **kwargs,
                )

                # 读取生成的输出文件
                content_list, _ = self._read_output_files(
                    base_output_dir, name_without_suff, method="ocr"
                )
                return content_list

            except MineruExecutionError:
                raise

            finally:
                # 如果创建了临时转换文件,则清理
                if temp_converted_file and temp_converted_file.exists():
                    try:
                        temp_converted_file.unlink()
                        temp_converted_file.parent.rmdir()  # 如果为空则删除临时目录
                    except Exception:
                        pass  # 忽略清理错误

        except Exception as e:
            logging.error(f"Error in parse_image: {str(e)}")
            raise

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        通过先转换为 PDF,然后使用 MinerU 2.0 解析的方式解析 Office 文档

        注意: 此方法需要单独安装 LibreOffice 以进行 PDF 转换。
        MinerU 2.0 不再包含内置的 Office 文档转换功能。

        支持的格式: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: 文档文件的路径 (.doc, .docx, .ppt, .pptx, .xls, .xlsx)
            output_dir: 输出目录路径
            lang: 用于 OCR 优化的文档语言
            **kwargs: mineru 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        try:
            # 使用基类方法将 Office 文档转换为 PDF
            pdf_path = self.convert_office_to_pdf(doc_path, output_dir)

            # 解析转换后的 PDF
            return self.parse_pdf(
                pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
            )

        except Exception as e:
            logging.error(f"Error in parse_office_doc: {str(e)}")
            raise

    def parse_text_file(
        self,
        text_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        通过先转换为 PDF,然后使用 MinerU 2.0 解析的方式解析文本文件

        支持的格式: .txt, .md

        Args:
            text_path: 文本文件的路径 (.txt, .md)
            output_dir: 输出目录路径
            lang: 用于 OCR 优化的文档语言
            **kwargs: mineru 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        try:
            # 使用基类方法将文本文件转换为 PDF
            pdf_path = self.convert_text_to_pdf(text_path, output_dir)

            # 解析转换后的 PDF
            return self.parse_pdf(
                pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
            )

        except Exception as e:
            logging.error(f"Error in parse_text_file: {str(e)}")
            raise

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        根据文件扩展名使用 MinerU 2.0 解析文档

        Args:
            file_path: 要解析的文件路径
            method: 解析方法 (auto, txt, ocr)
            output_dir: 输出目录路径
            lang: 用于 OCR 优化的文档语言
            **kwargs: mineru 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        # 转换为 Path 对象
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # 获取文件扩展名
        ext = file_path.suffix.lower()

        # 根据文件类型选择适当的解析器
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.IMAGE_FORMATS:
            return self.parse_image(file_path, output_dir, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            logging.warning(
                f"Warning: Office document detected ({ext}). "
                f"MinerU 2.0 requires conversion to PDF first."
            )
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.TEXT_FORMATS:
            return self.parse_text_file(file_path, output_dir, lang, **kwargs)
        else:
            # 对于不支持的文件类型,尝试作为 PDF 解析
            logging.warning(
                f"Warning: Unsupported file extension '{ext}', "
                f"attempting to parse as PDF"
            )
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)

    def check_installation(self) -> bool:
        """
        检查 MinerU 2.0 是否正确安装

        Returns:
            bool: 如果安装有效则为 True,否则为 False
        """
        try:
            # 准备子进程参数以在 Windows 上隐藏控制台窗口
            import platform

            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # 在 Windows 上隐藏控制台窗口
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(["mineru", "--version"], **subprocess_kwargs)
            logging.debug(f"MinerU version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.debug(
                "MinerU 2.0 is not properly installed. "
                "Please install it using: pip install -U 'mineru[core]'"
            )
            return False


class DoclingParser(Parser):
    """
    Docling 文档解析工具类。

    专门用于解析 Office 文档和 HTML 文件,将内容转换为结构化数据
    并生成 markdown 和 JSON 输出。
    """

    # 定义 Docling 特定的格式
    HTML_FORMATS = {".html", ".htm", ".xhtml"}

    def __init__(self) -> None:
        """初始化 DoclingParser"""
        super().__init__()

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        使用 Docling 解析 PDF 文档

        Args:
            pdf_path: PDF 文件的路径
            output_dir: 输出目录路径
            method: 解析方法 (auto, txt, ocr)
            lang: 用于 OCR 优化的文档语言
            **kwargs: docling 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        try:
            # 转换为 Path 对象以便于处理
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # 准备输出目录
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # 运行 docling 命令
            self._run_docling_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # 读取生成的输出文件
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        根据文件扩展名使用 Docling 解析文档

        Args:
            file_path: 要解析的文件路径
            method: 解析方法
            output_dir: 输出目录路径
            lang: 用于优化的文档语言
            **kwargs: docling 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        # 转换为 Path 对象
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # 获取文件扩展名
        ext = file_path.suffix.lower()

        # 根据文件类型选择适当的解析器
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir, lang, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Docling only supports PDF files, Office formats ({', '.join(self.OFFICE_FORMATS)}) "
                f"and HTML formats ({', '.join(self.HTML_FORMATS)})"
            )

    def _run_docling_command(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        file_stem: str,
        **kwargs,
    ) -> None:
        """
        运行 docling 命令行工具

        Args:
            input_path: 输入文件或目录的路径
            output_dir: 输出目录路径
            file_stem: 用于创建子目录的文件名(不含扩展名)
            **kwargs: docling 命令的附加参数
        """
        # 创建类似于 MinerU 的子目录结构
        file_output_dir = Path(output_dir) / file_stem / "docling"
        file_output_dir.mkdir(parents=True, exist_ok=True)

        cmd_json = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "json",
            str(input_path),
        ]
        cmd_md = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "md",
            str(input_path),
        ]

        try:
            # 准备子进程参数以在 Windows 上隐藏控制台窗口
            import platform

            docling_subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # 在 Windows 上隐藏控制台窗口
            if platform.system() == "Windows":
                docling_subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result_json = subprocess.run(cmd_json, **docling_subprocess_kwargs)
            result_md = subprocess.run(cmd_md, **docling_subprocess_kwargs)
            logging.info("Docling command executed successfully")
            if result_json.stdout:
                logging.debug(f"JSON cmd output: {result_json.stdout}")
            if result_md.stdout:
                logging.debug(f"Markdown cmd output: {result_md.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running docling command: {e}")
            if e.stderr:
                logging.error(f"Error details: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "docling command not found. Please ensure Docling is properly installed."
            )

    def _read_output_files(
        self,
        output_dir: Path,
        file_stem: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        读取由 docling 生成的输出文件并转换为 MinerU 格式

        Args:
            output_dir: 输出目录
            file_stem: 不带扩展名的文件名

        Returns:
            包含 (内容列表 JSON, Markdown 文本) 的元组
        """
        # 使用类似于 MinerU 的子目录结构
        file_subdir = output_dir / file_stem / "docling"
        md_file = file_subdir / f"{file_stem}.md"
        json_file = file_subdir / f"{file_stem}.json"

        # 读取 markdown 内容
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logging.warning(f"Could not read markdown file {md_file}: {e}")

        # 读取 JSON 内容并转换格式
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    docling_content = json.load(f)
                    # 将 docling 格式转换为 minerU 格式
                    content_list = self.read_from_block_recursive(
                        docling_content["body"],
                        "body",
                        file_subdir,
                        0,
                        "0",
                        docling_content,
                    )
            except Exception as e:
                logging.warning(f"Could not read or convert JSON file {json_file}: {e}")
        return content_list, md_content

    def read_from_block_recursive(
        self,
        block,
        type: str,
        output_dir: Path,
        cnt: int,
        num: str,
        docling_content: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content_list = []
        if not block.get("children"):
            cnt += 1
            content_list.append(self.read_from_block(block, type, output_dir, cnt, num))
        else:
            if type not in ["groups", "body"]:
                cnt += 1
                content_list.append(
                    self.read_from_block(block, type, output_dir, cnt, num)
                )
            members = block["children"]
            for member in members:
                cnt += 1
                member_tag = member["$ref"]
                member_type = member_tag.split("/")[1]
                member_num = member_tag.split("/")[2]
                member_block = docling_content[member_type][int(member_num)]
                content_list.extend(
                    self.read_from_block_recursive(
                        member_block,
                        member_type,
                        output_dir,
                        cnt,
                        member_num,
                        docling_content,
                    )
                )
        return content_list

    def read_from_block(
        self, block, type: str, output_dir: Path, cnt: int, num: str
    ) -> Dict[str, Any]:
        if type == "texts":
            if block["label"] == "formula":
                return {
                    "type": "equation",
                    "img_path": "",
                    "text": block["orig"],
                    "text_format": "unknown",
                    "page_idx": cnt // 10,
                }
            else:
                return {
                    "type": "text",
                    "text": block["orig"],
                    "page_idx": cnt // 10,
                }
        elif type == "pictures":
            try:
                base64_uri = block["image"]["uri"]
                base64_str = base64_uri.split(",")[1]
                # Create images directory within the docling subdirectory
                image_dir = output_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                image_path = image_dir / f"image_{num}.png"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(base64_str))
                return {
                    "type": "image",
                    "img_path": str(image_path.resolve()),  # Convert to absolute path
                    "image_caption": block.get("caption", ""),
                    "image_footnote": block.get("footnote", ""),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process image {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Image processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }
        else:
            try:
                return {
                    "type": "table",
                    "img_path": "",
                    "table_caption": block.get("caption", ""),
                    "table_footnote": block.get("footnote", ""),
                    "table_body": block.get("data", []),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process table {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Table processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        直接使用 Docling 解析 Office 文档

        支持的格式: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: 文档文件的路径
            output_dir: 输出目录路径
            lang: 用于优化的文档语言
            **kwargs: docling 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        try:
            # 转换为 Path 对象
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Document file does not exist: {doc_path}")

            if doc_path.suffix.lower() not in self.OFFICE_FORMATS:
                raise ValueError(f"Unsupported office format: {doc_path.suffix}")

            name_without_suff = doc_path.stem

            # 准备输出目录
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # 运行 docling 命令
            self._run_docling_command(
                input_path=doc_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # 读取生成的输出文件
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_office_doc: {str(e)}")
            raise

    def parse_html(
        self,
        html_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        使用 Docling 解析 HTML 文档

        支持的格式: .html, .htm, .xhtml

        Args:
            html_path: HTML 文件的路径
            output_dir: 输出目录路径
            lang: 用于优化的文档语言
            **kwargs: docling 命令的附加参数

        Returns:
            List[Dict[str, Any]]: 内容块列表
        """
        try:
            # 转换为 Path 对象
            html_path = Path(html_path)
            if not html_path.exists():
                raise FileNotFoundError(f"HTML file does not exist: {html_path}")

            if html_path.suffix.lower() not in self.HTML_FORMATS:
                raise ValueError(f"Unsupported HTML format: {html_path.suffix}")

            name_without_suff = html_path.stem

            # 准备输出目录
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = html_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # 运行 docling 命令
            self._run_docling_command(
                input_path=html_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # 读取生成的输出文件
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_html: {str(e)}")
            raise

    def check_installation(self) -> bool:
        """
        检查 Docling 是否正确安装

        Returns:
            bool: 如果安装有效则为 True,否则为 False
        """
        try:
            # 准备子进程参数以在 Windows 上隐藏控制台窗口
            import platform

            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # 在 Windows 上隐藏控制台窗口
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(["docling", "--version"], **subprocess_kwargs)
            logging.debug(f"Docling version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.debug(
                "Docling is not properly installed. "
                "Please ensure it is installed correctly."
            )
            return False


def main():
    """
    从命令行运行文档解析器的主函数
    """
    parser = argparse.ArgumentParser(
        description="使用 MinerU 2.0 或 Docling 解析文档"
    )
    parser.add_argument("file_path", help="要解析的文档路径")
    parser.add_argument("--output", "-o", help="输出目录路径")
    parser.add_argument(
        "--method",
        "-m",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="解析方法 (auto, txt, ocr)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        help="用于 OCR 优化的文档语言 (例如: ch, en, ja)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=[
            "pipeline",
            "vlm-transformers",
            "vlm-sglang-engine",
            "vlm-sglang-client",
        ],
        default="pipeline",
        help="解析后端",
    )
    parser.add_argument(
        "--device",
        "-d",
        help="推理设备 (例如: cpu, cuda, cuda:0, npu, mps)",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope", "local"],
        default="huggingface",
        help="模型源",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help="禁用公式解析",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="禁用表格解析",
    )
    parser.add_argument(
        "--stats", action="store_true", help="显示内容统计信息"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="检查解析器安装",
    )
    parser.add_argument(
        "--parser",
        choices=["mineru", "docling"],
        default="mineru",
        help="解析器选择",
    )
    parser.add_argument(
        "--vlm_url",
        help="当后端为 `vlm-sglang-client` 时,需要指定 server_url,例如: `http://127.0.0.1:30000`",
    )

    args = parser.parse_args()

    # 如果请求,检查安装
    if args.check:
        doc_parser = DoclingParser() if args.parser == "docling" else MineruParser()
        if doc_parser.check_installation():
            print(f"✅ {args.parser.title()} is properly installed")
            return 0
        else:
            print(f"❌ {args.parser.title()} installation check failed")
            return 1

    try:
        # 解析文档
        doc_parser = DoclingParser() if args.parser == "docling" else MineruParser()
        content_list = doc_parser.parse_document(
            file_path=args.file_path,
            method=args.method,
            output_dir=args.output,
            lang=args.lang,
            backend=args.backend,
            device=args.device,
            source=args.source,
            formula=not args.no_formula,
            table=not args.no_table,
            vlm_url=args.vlm_url,
        )

        print(f"✅ Successfully parsed: {args.file_path}")
        print(f"📊 Extracted {len(content_list)} content blocks")

        # 如果请求,显示统计信息
        if args.stats:
            print("\n📈 Document Statistics:")
            print(f"Total content blocks: {len(content_list)}")

            # 统计不同类型的内容
            content_types = {}
            for item in content_list:
                if isinstance(item, dict):
                    content_type = item.get("type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            if content_types:
                print("\n📋 Content Type Distribution:")
                for content_type, count in sorted(content_types.items()):
                    print(f"  • {content_type}: {count}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
