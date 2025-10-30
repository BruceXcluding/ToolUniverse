#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA工具包CLI入口 - 支持四个生物工具的命令行界面
"""
import os
import sys
import json
import argparse
import traceback

# 添加项目根目录和src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

def run_rna_fold(args):
    """运行RNA二级结构预测工具"""
    try:
        from tooluniverse.bio_models.tools.rna_fold_tool import RNAFoldTool
        
        rna_fold_tool = RNAFoldTool()
        
        # 转换参数格式，调用analyze方法
        result = rna_fold_tool.analyze(args.sequence, "structure_prediction")
        
        if args.json:
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            # 非JSON模式下输出友好格式
            if result.get("status") == "success":
                structure = result.get("structure", "")
                energy = result.get("energy", "")
                sys.stdout.write(f"序列: {args.sequence}\n")
                sys.stdout.write(f"结构: {structure}\n")
                if energy:
                    sys.stdout.write(f"自由能: {energy}\n")
            else:
                sys.stdout.write(f"预测失败: {result.get('error', '未知错误')}\n")
        
        return 0
    except Exception as e:
        sys.stderr.write(f"错误: {str(e)}\n")
        return 1

def run_seq_compare(args):
    """运行序列比较工具"""
    try:
        from tooluniverse.bio_models.tools.seq_compare_tool import SeqCompareTool
        
        seq_compare_tool = SeqCompareTool()
        
        # 使用compare_single_pair方法
        result = seq_compare_tool.compare_single_pair(args.seq1, args.seq2)
        
        if args.json:
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            # 非JSON模式下输出友好格式
            if result.get("status") == "success":
                identity = result.get("identity", 0)
                mismatch_count = result.get("mismatch_count", 0)
                sys.stdout.write(f"序列1: {args.seq1}\n")
                sys.stdout.write(f"序列2: {args.seq2}\n")
                sys.stdout.write(f"相似度: {identity:.2f}%\n")
                sys.stdout.write(f"不匹配数: {mismatch_count}\n")
            else:
                sys.stdout.write(f"比较失败: {result.get('error', '未知错误')}\n")
        
        return 0
    except Exception as e:
        sys.stderr.write(f"错误: {str(e)}\n")
        return 1

def run_jaspar_scan(args):
    """运行JASPAR Motif扫描工具"""
    try:
        from tooluniverse.bio_models.tools.jaspar_scan_tool import JasparScanTool
        
        jaspar_tool = JasparScanTool()
        
        # 如果是JSON模式，设置为静默模式
        if args.json:
            jaspar_tool.quiet_mode = True
        
        # 调用analyze方法，传入参数
        scan_result = jaspar_tool.analyze(
            args.sequence, 
            "motif_detection", 
            species=args.species,
            threshold_ratio=args.threshold,
            top_n=args.top
        )
        
        # 正确获取结果中的motif hits
        # 从scan_result['results'][0]['motif_hits']获取匹配结果
        hits = []
        if scan_result.get('success') and 'results' in scan_result and len(scan_result['results']) > 0:
            hits = scan_result['results'][0].get('motif_hits', [])
        
        if args.json:
            # 转换结果格式以匹配CLI要求
            formatted_results = []
            for hit in hits:
                formatted_results.append({
                    "matrix_id": hit.get("matrix_id", ""),
                    "name": hit.get("name", ""),
                    "start": hit.get("start", 0),
                    "end": hit.get("end", 0),
                    "score": hit.get("score", 0),
                    "kmer": hit.get("sequence", "")
                })
            sys.stdout.write(json.dumps(formatted_results, ensure_ascii=False, indent=2))
        else:
            # 非JSON模式下输出表格格式
            header = f"{'Matrix_ID':<10} {'Name':<15} {'Start':>6} {'End':>6} {'Score':>8} {'K-mer'}"
            sys.stdout.write(header + "\n")
            sys.stdout.write("-" * 60 + "\n")
            if not hits:
                sys.stdout.write("未找到匹配的motif\n")
            for hit in hits:
                sys.stdout.write(f"{hit.get('matrix_id', ''):<10} {hit.get('name', ''):<15} {hit.get('start', 0):6d} {hit.get('end', 0):6d} {hit.get('score', 0):8.2f} {hit.get('sequence', '')}\n")
        
        return 0
    except Exception as e:
        sys.stderr.write(f"错误: {str(e)}\n")
        return 1

def run_blast_search(args):
    """运行BLAST搜索工具"""
    try:
        from tooluniverse.bio_models.tools.blast_search_tool import BlastSearchTool
        
        blast_tool = BlastSearchTool()
        
        if args.parse_xml:
            # 解析XML文件
            results = blast_tool.parse_xml_result(args.parse_xml, max_results=args.max_results)
        else:
            # 执行搜索
            results = blast_tool.analyze(
                args.sequence,
                "function_annotation",
                program=args.program,
                database=args.database,
                email=args.email,
                max_results=args.max_results,
                expect_threshold=args.expect
            )
        
        if args.json:
            sys.stdout.write(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            # 非JSON模式下输出表格格式
            if not results:
                sys.stdout.write("未找到匹配结果\n")
                return 0
            
            header = f"{'Subject':<50} {'E-value':>12} {'Score':>8} {'Identity':>10}"
            sys.stdout.write(header + "\n")
            sys.stdout.write("-" * 80 + "\n")
            
            for r in results:
                subject = r.get('subject_title', '')[:48]
                sys.stdout.write(f"{subject:<50} {r.get('expect', 0):>12.2e} {r.get('score', 0):>8.1f} {r.get('identities', ''):>10}\n")
        
        return 0
    except Exception as e:
        sys.stderr.write(f"错误: {str(e)}\n")
        return 1

def main():
    """主函数，解析命令行参数并调用相应工具"""
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description="RNA工具包CLI - 支持RNA二级结构预测、序列比较、Motif扫描和BLAST搜索",
        usage="%(prog)s <command> [options]"
    )
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # RNA折叠命令
    rna_fold_parser = subparsers.add_parser("rna_fold", description="RNA 二级结构预测工具")
    rna_fold_parser.add_argument("--sequence", required=True, help="RNA 序列")
    rna_fold_parser.add_argument("--json", action="store_true", help="输出 JSON 格式结果")
    rna_fold_parser.set_defaults(func=run_rna_fold)
    
    # 序列比较命令
    seq_compare_parser = subparsers.add_parser("seq_compare", description="Point-to-point comparison of two equal-length sequences.")
    seq_compare_parser.add_argument("--seq1", required=True, help="First sequence.")
    seq_compare_parser.add_argument("--seq2", required=True, help="Second sequence.")
    seq_compare_parser.add_argument("--json", action="store_true", help="Output result in JSON format.")
    seq_compare_parser.set_defaults(func=run_seq_compare)
    
    # JASPAR扫描命令
    jaspar_scan_parser = subparsers.add_parser("jaspar_scan", description="Scan motifs using JASPAR PWM database.")
    jaspar_scan_parser.add_argument("--sequence", required=True, help="DNA/RNA sequence (U auto-converted to T).")
    jaspar_scan_parser.add_argument("--species", type=int, default=9606, help="NCBI species ID, default 9606 (human).")
    jaspar_scan_parser.add_argument("--threshold", type=float, default=0.85, help="Threshold ratio, default 0.85.")
    jaspar_scan_parser.add_argument("--top", type=int, default=10, help="Number of top hits to return.")
    jaspar_scan_parser.add_argument("--json", action="store_true", help="Output in JSON format.")
    jaspar_scan_parser.set_defaults(func=run_jaspar_scan)
    
    # BLAST搜索命令
    blast_search_parser = subparsers.add_parser("blast_search", description="NCBI BLAST 在线搜索工具")
    blast_search_parser.add_argument("--sequence", required=True, help="DNA/RNA/蛋白质序列")
    blast_search_parser.add_argument("--program", type=str, default="blastn", 
                        choices=["blastn", "blastp", "blastx", "tblastn", "tblastx"],
                        help="BLAST 程序类型，默认 blastn")
    blast_search_parser.add_argument("--database", type=str, default="nt",
                        help="搜索数据库，默认 nt (核苷酸库)")
    blast_search_parser.add_argument("--email", type=str, default="user@example.com",
                        help="NCBI 邮箱地址，默认 user@example.com")
    blast_search_parser.add_argument("--max-results", type=int, default=10,
                        help="返回的最大结果数，默认 10")
    blast_search_parser.add_argument("--expect", type=float, default=0.001,
                        help="E值阈值，默认 0.001")
    blast_search_parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    blast_search_parser.add_argument("--parse-xml", type=str, help="解析已保存的 XML 文件")
    blast_search_parser.set_defaults(func=run_blast_search)
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行对应命令
    sys.exit(args.func(args))

if __name__ == "__main__":
    main()