"""
批量和并行文档解析

此模块提供并行处理多个文档的功能，
包括进度报告和错误处理。
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from tqdm import tqdm

from .parser import MineruParser, DoclingParser


@dataclass
class BatchProcessingResult:
    """批处理操作的结果"""

    successful_files: List[str]
    failed_files: List[str]
    total_files: int
    processing_time: float
    errors: Dict[str, str]
    output_dir: str

    @property
    def success_rate(self) -> float:
        """计算成功率（百分比）"""
        if self.total_files == 0:
            return 0.0
        return (len(self.successful_files) / self.total_files) * 100

    def summary(self) -> str:
        """生成批处理结果的摘要"""
        return (
            f"Batch Processing Summary:\n"
            f"  Total files: {self.total_files}\n"
            f"  Successful: {len(self.successful_files)} ({self.success_rate:.1f}%)\n"
            f"  Failed: {len(self.failed_files)}\n"
            f"  Processing time: {self.processing_time:.2f} seconds\n"
            f"  Output directory: {self.output_dir}"
        )


class BatchParser:
    """
    具有并行处理能力的批量文档解析器

    支持并发处理多个文档，具有进度跟踪
    和全面的错误处理功能。
    """

    def __init__(
        self,
        parser_type: str = "mineru",
        max_workers: int = 4,
        show_progress: bool = True,
        timeout_per_file: int = 300,
        skip_installation_check: bool = False,
    ):
        """
        初始化批处理解析器

        参数:
            parser_type: 要使用的解析器类型（"mineru" 或 "docling"）
            max_workers: 最大并行工作线程数
            show_progress: 是否显示进度条
            timeout_per_file: 每个文件的超时时间（秒）
            skip_installation_check: 跳过解析器安装检查（对测试有用）
        """
        self.parser_type = parser_type
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.timeout_per_file = timeout_per_file
        self.logger = logging.getLogger(__name__)

        # 初始化解析器
        if parser_type == "mineru":
            self.parser = MineruParser()
        elif parser_type == "docling":
            self.parser = DoclingParser()
        else:
            raise ValueError(f"Unsupported parser type: {parser_type}")

        # 检查解析器安装（可选）
        if not skip_installation_check:
            if not self.parser.check_installation():
                self.logger.warning(
                    f"{parser_type.title()} parser installation check failed. "
                    f"This may be due to package conflicts. "
                    f"Use skip_installation_check=True to bypass this check."
                )
                # 不抛出错误，只是警告 - 解析器可能仍然可以工作

    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名列表"""
        return list(
            self.parser.OFFICE_FORMATS
            | self.parser.IMAGE_FORMATS
            | self.parser.TEXT_FORMATS
            | {".pdf"}
        )

    def filter_supported_files(
        self, file_paths: List[str], recursive: bool = True
    ) -> List[str]:
        """
        过滤文件路径，只包含支持的文件类型

        参数:
            file_paths: 文件路径或目录列表
            recursive: 是否递归搜索目录

        返回:
            支持的文件路径列表
        """
        supported_extensions = set(self.get_supported_extensions())
        supported_files = []

        for path_str in file_paths:
            path = Path(path_str)

            if path.is_file():
                if path.suffix.lower() in supported_extensions:
                    supported_files.append(str(path))
                else:
                    self.logger.warning(f"Unsupported file type: {path}")

            elif path.is_dir():
                if recursive:
                    # 递归查找所有文件
                    for file_path in path.rglob("*"):
                        if (
                            file_path.is_file()
                            and file_path.suffix.lower() in supported_extensions
                        ):
                            supported_files.append(str(file_path))
                else:
                    # 仅目录中的文件（不包括子目录）
                    for file_path in path.glob("*"):
                        if (
                            file_path.is_file()
                            and file_path.suffix.lower() in supported_extensions
                        ):
                            supported_files.append(str(file_path))

            else:
                self.logger.warning(f"Path does not exist: {path}")

        return supported_files

    def process_single_file(
        self, file_path: str, output_dir: str, parse_method: str = "auto", **kwargs
    ) -> Tuple[bool, str, Optional[str]]:
        """
        处理单个文件

        参数:
            file_path: 要处理的文件路径
            output_dir: 输出目录
            parse_method: 解析方法
            **kwargs: 额外的解析器参数

        返回:
            元组 (success, file_path, error_message)
        """
        try:
            start_time = time.time()

            # 创建文件特定的输出目录
            file_name = Path(file_path).stem
            file_output_dir = Path(output_dir) / file_name
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # 解析文档
            content_list = self.parser.parse_document(
                file_path=file_path,
                output_dir=str(file_output_dir),
                method=parse_method,
                **kwargs,
            )

            processing_time = time.time() - start_time

            self.logger.info(
                f"Successfully processed {file_path} "
                f"({len(content_list)} content blocks, {processing_time:.2f}s)"
            )

            return True, file_path, None

        except Exception as e:
            error_msg = f"Failed to process {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return False, file_path, error_msg

    def process_batch(
        self,
        file_paths: List[str],
        output_dir: str,
        parse_method: str = "auto",
        recursive: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        并行处理多个文件

        参数:
            file_paths: 要处理的文件路径或目录列表
            output_dir: 基础输出目录
            parse_method: 所有文件的解析方法
            recursive: 是否递归搜索目录
            **kwargs: 额外的解析器参数

        返回:
            包含处理统计信息的 BatchProcessingResult
        """
        start_time = time.time()

        # 过滤到支持的文件
        supported_files = self.filter_supported_files(file_paths, recursive)

        if not supported_files:
            self.logger.warning("No supported files found to process")
            return BatchProcessingResult(
                successful_files=[],
                failed_files=[],
                total_files=0,
                processing_time=0.0,
                errors={},
                output_dir=output_dir,
            )

        self.logger.info(f"Found {len(supported_files)} files to process")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 并行处理文件
        successful_files = []
        failed_files = []
        errors = {}

        # 如果需要，创建进度条
        pbar = None
        if self.show_progress:
            pbar = tqdm(
                total=len(supported_files),
                desc=f"Processing files ({self.parser_type})",
                unit="file",
            )

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(
                        self.process_single_file,
                        file_path,
                        output_dir,
                        parse_method,
                        **kwargs,
                    ): file_path
                    for file_path in supported_files
                }

                # 处理已完成的任务
                for future in as_completed(
                    future_to_file, timeout=self.timeout_per_file
                ):
                    success, file_path, error_msg = future.result()

                    if success:
                        successful_files.append(file_path)
                    else:
                        failed_files.append(file_path)
                        errors[file_path] = error_msg

                    if pbar:
                        pbar.update(1)

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            # 将剩余文件标记为失败
            for future in future_to_file:
                if not future.done():
                    file_path = future_to_file[future]
                    failed_files.append(file_path)
                    errors[file_path] = f"Processing interrupted: {str(e)}"
                    if pbar:
                        pbar.update(1)

        finally:
            if pbar:
                pbar.close()

        processing_time = time.time() - start_time

        # 创建结果
        result = BatchProcessingResult(
            successful_files=successful_files,
            failed_files=failed_files,
            total_files=len(supported_files),
            processing_time=processing_time,
            errors=errors,
            output_dir=output_dir,
        )

        # 记录摘要
        self.logger.info(result.summary())

        return result

    async def process_batch_async(
        self,
        file_paths: List[str],
        output_dir: str,
        parse_method: str = "auto",
        recursive: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        批处理的异步版本

        参数:
            file_paths: 要处理的文件路径或目录列表
            output_dir: 基础输出目录
            parse_method: 所有文件的解析方法
            recursive: 是否递归搜索目录
            **kwargs: 额外的解析器参数

        返回:
            包含处理统计信息的 BatchProcessingResult
        """
        # 在线程池中运行同步版本
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_batch,
            file_paths,
            output_dir,
            parse_method,
            recursive,
            **kwargs,
        )


def main():
    """批处理解析的命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch document parsing")
    parser.add_argument("paths", nargs="+", help="File paths or directories to process")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--parser",
        choices=["mineru", "docling"],
        default="mineru",
        help="Parser to use",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="Parsing method",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search directories recursively",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout per file (seconds)"
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # 创建批处理解析器
        batch_parser = BatchParser(
            parser_type=args.parser,
            max_workers=args.workers,
            show_progress=not args.no_progress,
            timeout_per_file=args.timeout,
        )

        # 处理文件
        result = batch_parser.process_batch(
            file_paths=args.paths,
            output_dir=args.output,
            parse_method=args.method,
            recursive=args.recursive,
        )

        # 打印摘要
        print("\n" + result.summary())

        # 如果有文件失败，以错误代码退出
        if result.failed_files:
            return 1

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
