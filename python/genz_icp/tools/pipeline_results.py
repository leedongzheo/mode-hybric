from dataclasses import dataclass
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table


class PipelineResults:
    def __init__(self) -> None:
        self._results = []
        # [THÊM MỚI] Danh sách lưu trữ thời gian cho từng frame
        self._breakdowns = []

    def empty(self) -> bool:
        return len(self._results) == 0

    def print(self) -> None:
        self.log_to_console()

    def append(self, desc: str, units: str, value: float, trunc: bool = False):
        @dataclass
        class Metric:
            desc: str
            units: str
            value: float

        self._results.append(Metric(desc, units, int(value) if trunc else value))

    # === [THÊM MỚI] Hàm để đẩy dữ liệu thời gian vào từ pipeline.py ===
    def append_breakdown(self, search_time: float, pca_time: float, opt_time: float):
        self._breakdowns.append((search_time, pca_time, opt_time))
    # ===================================================================

    def log_to_file(self, filename: str, title: Optional[str]) -> None:
        with open(filename, "wt") as logfile:
            console = Console(file=logfile, width=100, force_jupyter=False)
            if title:
                console.rule(title)
            console.print(self._rich_table(table_format=box.ASCII_DOUBLE_HEAD))
            
            # [THÊM MỚI] In bảng Breakdown vào file log
            if self._breakdowns:
                console.print()
                console.print(self._rich_breakdown_table(table_format=box.ASCII_DOUBLE_HEAD))

    def log_to_console(self) -> None:
        if self.empty():
            return
        console = Console()
        console.print(self._rich_table())
        
        # [THÊM MỚI] In bảng Breakdown ra màn hình Console
        if self._breakdowns:
            console.print()
            console.print(self._rich_breakdown_table())

    def _rich_table(self, table_format: box.Box = box.HORIZONTALS) -> Table:
        table = Table(box=table_format)
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", justify="center", style="magenta")
        table.add_column("Units", justify="left", style="green")
        for result in self._results:
            table.add_row(
                result.desc,
                f"{result.value:{'.3f' if isinstance(result.value, float) else ''}}",
                result.units,
            )
        return table

    # === [THÊM MỚI] Hàm tạo bảng tính trung bình Breakdown ===
    def _rich_breakdown_table(self, table_format: box.Box = box.HORIZONTALS) -> Table:
        # Tiêu đề bảng màu vàng nổi bật
        table = Table(box=table_format, title="Front-End Runtime Breakdown (Average)", title_style="bold yellow")
        table.add_column("Module", justify="right", style="cyan")
        table.add_column("Avg Time (ms)", justify="center", style="magenta")
        table.add_column("Percentage", justify="right", style="green")
        
        num_frames = len(self._breakdowns)
        avg_search = sum(b[0] for b in self._breakdowns) / num_frames
        avg_pca = sum(b[1] for b in self._breakdowns) / num_frames
        avg_opt = sum(b[2] for b in self._breakdowns) / num_frames
        
        total_avg = avg_search + avg_pca + avg_opt
        if total_avg == 0: total_avg = 1e-9 # Tránh chia cho 0 nếu bị lỗi logic
        
        table.add_row("Neighbor Search", f"{avg_search:.3f}", f"{(avg_search/total_avg)*100:.1f} %")
        table.add_row("PCA & Planarity", f"{avg_pca:.3f}", f"{(avg_pca/total_avg)*100:.1f} %")
        table.add_row("ICP Optimizer", f"{avg_opt:.3f}", f"{(avg_opt/total_avg)*100:.1f} %")
        
        # Thêm vạch kẻ ngang phân cách tổng
        table.add_section() 
        table.add_row("[bold]Total Front-End", f"[bold]{total_avg:.3f}", "[bold]100.0 %")
        
        return table
