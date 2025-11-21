"""
RAGAnything 的批处理功能

包含用于批量处理多个文档的方法
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import time

from .batch_parser import BatchParser, BatchProcessingResult

if TYPE_CHECKING:
    from .config import RAGAnythingConfig


class BatchMixin:
    """包含 RAGAnything 批处理功能的 BatchMixin 类"""

    # Mixin 属性的类型提示（混入 RAGAnything 时可用）
    config: "RAGAnythingConfig"
    logger: logging.Logger

    # 来自其他 mixins 的可用方法的类型提示
    async def _ensure_lightrag_initialized(self) -> None: ...
    async def process_document_complete(self, file_path: str, **kwargs) -> None: ...

    # ==========================================
    # 原始批处理方法（已恢复）
    # ==========================================

    async def process_folder_complete(
        self,
        folder_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = None,
        max_workers: int = None,
    ):
        """
        处理文件夹中所有支持的文件

        参数:
            folder_path: 包含待处理文件的文件夹路径
            output_dir: 解析输出目录（可选）
            parse_method: 要使用的解析方法（可选）
            display_stats: 是否显示统计信息（可选）
            split_by_character: 用于分割的字符（可选）
            split_by_character_only: 是否仅按字符分割（可选）
            file_extensions: 要处理的文件扩展名列表（可选）
            recursive: 是否递归处理文件夹（可选）
            max_workers: 并发处理的最大工作线程数（可选）
        """
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = True
        if file_extensions is None:
            file_extensions = self.config.supported_file_extensions
        if recursive is None:
            recursive = self.config.recursive_folder_processing
        if max_workers is None:
            max_workers = self.config.max_concurrent_files

        await self._ensure_lightrag_initialized()

        # 获取文件夹中的所有文件
        folder_path_obj = Path(folder_path)
        if not folder_path_obj.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # 根据支持的扩展名收集文件
        files_to_process = []
        for file_ext in file_extensions:
            if recursive:
                pattern = f"**/*{file_ext}"
            else:
                pattern = f"*{file_ext}"
            files_to_process.extend(folder_path_obj.glob(pattern))

        if not files_to_process:
            self.logger.warning(f"No supported files found in {folder_path}")
            return

        self.logger.info(
            f"Found {len(files_to_process)} files to process in {folder_path}"
        )

        # 如果输出目录不存在则创建
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 以受控的并发方式处理文件
        semaphore = asyncio.Semaphore(max_workers)
        tasks = []

        async def process_single_file(file_path: Path):
            async with semaphore:
                try:
                    await self.process_document_complete(
                        str(file_path),
                        output_dir=output_dir,
                        parse_method=parse_method,
                        split_by_character=split_by_character,
                        split_by_character_only=split_by_character_only,
                    )
                    return True, str(file_path), None
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
                    return False, str(file_path), str(e)

        # 为所有文件创建任务
        for file_path in files_to_process:
            task = asyncio.create_task(process_single_file(file_path))
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        successful_files = []
        failed_files = []
        for result in results:
            if isinstance(result, Exception):
                failed_files.append(("unknown", str(result)))
            else:
                success, file_path, error = result
                if success:
                    successful_files.append(file_path)
                else:
                    failed_files.append((file_path, error))

        # 如果需要，显示统计信息
        if display_stats:
            self.logger.info("Processing complete!")
            self.logger.info(f"  Successful: {len(successful_files)} files")
            self.logger.info(f"  Failed: {len(failed_files)} files")
            if failed_files:
                self.logger.warning("Failed files:")
                for file_path, error in failed_files:
                    self.logger.warning(f"  - {file_path}: {error}")

    # ==========================================
    # 新的增强批处理方法
    # ==========================================

    def process_documents_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        使用新的 BatchParser 批量处理多个文档

        参数:
            file_paths: 要处理的文件路径或目录列表
            output_dir: 解析文件的输出目录
            parse_method: 要使用的解析方法
            max_workers: 并行处理的最大工作线程数
            recursive: 是否递归处理目录
            show_progress: 是否显示进度条
            **kwargs: 传递给解析器的额外参数

        返回:
            BatchProcessingResult: 批处理结果
        """
        # 如果未指定，使用配置默认值
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        # 创建批处理解析器
        batch_parser = BatchParser(
            parser_type=self.config.parser,
            max_workers=max_workers,
            show_progress=show_progress,
            skip_installation_check=True,  # 跳过安装检查以获得更好的用户体验
        )

        # 处理批次
        return batch_parser.process_batch(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            recursive=recursive,
            **kwargs,
        )

    async def process_documents_batch_async(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        异步批量处理多个文档

        参数:
            file_paths: 要处理的文件路径或目录列表
            output_dir: 解析文件的输出目录
            parse_method: 要使用的解析方法
            max_workers: 并行处理的最大工作线程数
            recursive: 是否递归处理目录
            show_progress: 是否显示进度条
            **kwargs: 传递给解析器的额外参数

        返回:
            BatchProcessingResult: 批处理结果
        """
        # 如果未指定，使用配置默认值
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        # 创建批处理解析器
        batch_parser = BatchParser(
            parser_type=self.config.parser,
            max_workers=max_workers,
            show_progress=show_progress,
            skip_installation_check=True,  # 跳过安装检查以获得更好的用户体验
        )

        # 异步处理批次
        return await batch_parser.process_batch_async(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            recursive=recursive,
            **kwargs,
        )

    def get_supported_file_extensions(self) -> List[str]:
        """获取批处理支持的文件扩展名列表"""
        batch_parser = BatchParser(parser_type=self.config.parser)
        return batch_parser.get_supported_extensions()

    def filter_supported_files(
        self, file_paths: List[str], recursive: Optional[bool] = None
    ) -> List[str]:
        """
        过滤文件路径，只包含支持的文件类型

        参数:
            file_paths: 要过滤的文件路径列表
            recursive: 是否递归处理目录

        返回:
            支持的文件路径列表
        """
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        batch_parser = BatchParser(parser_type=self.config.parser)
        return batch_parser.filter_supported_files(file_paths, recursive)

    async def process_documents_with_rag_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        批量处理文档，然后将它们添加到 RAG

        此方法结合了文档解析和 RAG 插入:
        1. 首先，使用批处理解析所有文档
        2. 然后，使用 RAG 处理每个成功解析的文档

        参数:
            file_paths: 要处理的文件路径或目录列表
            output_dir: 解析文件的输出目录
            parse_method: 要使用的解析方法
            max_workers: 并行处理的最大工作线程数
            recursive: 是否递归处理目录
            show_progress: 是否显示进度条
            **kwargs: 传递给解析器的额外参数

        返回:
            包含解析结果和 RAG 处理结果的字典
        """
        start_time = time.time()

        # 如果未指定，使用配置默认值
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        self.logger.info("Starting batch processing with RAG integration")

        # 步骤 1: 批量解析文档
        parse_result = self.process_documents_batch(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            max_workers=max_workers,
            recursive=recursive,
            show_progress=show_progress,
            **kwargs,
        )

        # 步骤 2: 使用 RAG 处理
        # 初始化 RAG 系统
        await self._ensure_lightrag_initialized()

        # 然后，使用 RAG 处理每个成功的文件
        rag_results = {}

        if parse_result.successful_files:
            self.logger.info(
                f"Processing {len(parse_result.successful_files)} files with RAG"
            )

            # 使用 RAG 处理文件（未来可以并行化）
            for file_path in parse_result.successful_files:
                try:
                    # 使用 RAG 处理成功解析的文件
                    await self.process_document_complete(
                        file_path,
                        output_dir=output_dir,
                        parse_method=parse_method,
                        **kwargs,
                    )

                    # 获取有关已处理内容的一些统计信息
                    # 这需要在 RAG 系统中进行额外的跟踪
                    rag_results[file_path] = {"status": "success", "processed": True}

                except Exception as e:
                    self.logger.error(
                        f"Failed to process {file_path} with RAG: {str(e)}"
                    )
                    rag_results[file_path] = {
                        "status": "failed",
                        "error": str(e),
                        "processed": False,
                    }

        processing_time = time.time() - start_time

        return {
            "parse_result": parse_result,
            "rag_results": rag_results,
            "total_processing_time": processing_time,
            "successful_rag_files": len(
                [r for r in rag_results.values() if r["processed"]]
            ),
            "failed_rag_files": len(
                [r for r in rag_results.values() if not r["processed"]]
            ),
        }
