"""
生成棋盘格标定板图片
- 适合 A4 纸打印
"""

import numpy as np


def generate_checkerboard(
    inner_corners_cols: int = 7,  # 内角点列数
    inner_corners_rows: int = 5,  # 内角点行数
    square_size_mm: float = 22.0,  # 每格边长（毫米）
    output_path: str = "checkerboard_7x5_22mm.pdf",
):
    """
    生成 A4 尺寸的棋盘格 PDF，棋盘格居中
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # A4 尺寸（横向）
    a4_width_mm = 297
    a4_height_mm = 210
    
    # 格子数量
    num_cols = inner_corners_cols + 1
    num_rows = inner_corners_rows + 1
    
    # 棋盘格尺寸
    board_width_mm = num_cols * square_size_mm
    board_height_mm = num_rows * square_size_mm
    
    # 居中偏移
    offset_x = (a4_width_mm - board_width_mm) / 2
    offset_y = (a4_height_mm - board_height_mm) / 2
    
    # 创建 A4 尺寸的图形
    fig_width = a4_width_mm / 25.4  # 英寸
    fig_height = a4_height_mm / 25.4
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, a4_width_mm)
    ax.set_ylim(0, a4_height_mm)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 白色背景（整个 A4）
    ax.add_patch(Rectangle((0, 0), a4_width_mm, a4_height_mm, 
                           facecolor='white', edgecolor='none'))
    
    # 绘制黑色格子（居中）
    for row in range(num_rows):
        for col in range(num_cols):
            if (row + col) % 2 == 0:
                x = offset_x + col * square_size_mm
                y = offset_y + (num_rows - 1 - row) * square_size_mm
                ax.add_patch(Rectangle((x, y), square_size_mm, square_size_mm,
                                       facecolor='black', edgecolor='none'))
    
    # 添加标注文字（左上角）
    text1 = f"{inner_corners_cols}x{inner_corners_rows} inner corners, {square_size_mm}mm/square"
    ax.text(5, a4_height_mm - 5, text1, fontsize=8, color='gray', va='top')
    
    # 添加尺寸标注线（帮助验证打印尺寸）
    # 在棋盘格下方画一条 50mm 的参考线
    ref_y = offset_y - 10
    ref_x_start = offset_x
    ref_x_end = offset_x + 50
    ax.plot([ref_x_start, ref_x_end], [ref_y, ref_y], 'k-', linewidth=1)
    ax.plot([ref_x_start, ref_x_start], [ref_y - 2, ref_y + 2], 'k-', linewidth=1)
    ax.plot([ref_x_end, ref_x_end], [ref_y - 2, ref_y + 2], 'k-', linewidth=1)
    ax.text((ref_x_start + ref_x_end) / 2, ref_y - 3, "50mm", 
            fontsize=8, ha='center', va='top')
    
    # 保存为 PDF（无边距）
    plt.savefig(output_path, format='pdf', 
                bbox_inches='tight', pad_inches=0,
                metadata={'Title': f'Checkerboard {inner_corners_cols}x{inner_corners_rows}'})
    plt.close()
    
    # 打印信息
    print("=" * 60)
    print("棋盘格标定板生成完成！")
    print("=" * 60)
    print(f"  文件: {output_path}")
    print(f"  内角点: {inner_corners_cols} × {inner_corners_rows}")
    print(f"  格子数: {num_cols} × {num_rows}")
    print(f"  每格尺寸: {square_size_mm}mm")
    print(f"  棋盘格尺寸: {board_width_mm}mm × {board_height_mm}mm")
    print(f"  PDF 尺寸: A4 横向 ({a4_width_mm}mm × {a4_height_mm}mm)")
    print("=" * 60)
    print("\n打印说明：")
    print("  1. 使用 A4 纸")
    print("  2. 打印设置选择「实际大小」或「100%」")
    print("  3. 【重要】关闭「适应页面」或「缩放」选项！")
    print(f"  4. 打印后用尺子测量下方的 50mm 参考线验证")
    print("=" * 60)
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成棋盘格标定板')
    parser.add_argument("--cols", type=int, default=9, help="内角点列数（默认7）")
    parser.add_argument("--rows", type=int, default=6, help="内角点行数（默认5）")
    parser.add_argument("--size", type=float, default=22.0, help="每格边长mm（默认22）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"checkerboard_{args.cols}x{args.rows}_{int(args.size)}mm.pdf"
    
    generate_checkerboard(
        inner_corners_cols=args.cols,
        inner_corners_rows=args.rows,
        square_size_mm=args.size,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()