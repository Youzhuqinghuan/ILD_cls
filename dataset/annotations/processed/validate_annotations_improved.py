#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版医学影像标注验证程序
基于分析报告的建议，修正了判别逻辑以提高验证准确率

主要改进：
1. 修正UIP判别逻辑：允许只有蜂窝影或只有网格影
2. 放宽NSIP判别条件：允许manifestation为subpleural的情况
3. 优化判别优先级
"""

import json
from typing import Dict, List, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def predict_disease_pattern_improved(abnormal_manifestation_presence: List[int], 
                                   overall_axial_distribution: str, 
                                   overall_manifestation: str) -> str:
    """
    改进版疾病模式预测函数
    
    Args:
        abnormal_manifestation_presence: [蜂窝, 网格, 实变, 磨玻璃]
        overall_axial_distribution: 轴向分布
        overall_manifestation: 表现形式
    
    Returns:
        预测的疾病模式
    """
    honeycombing = abnormal_manifestation_presence[0]  # 蜂窝影
    reticulation = abnormal_manifestation_presence[1]  # 网格影
    consolidation = abnormal_manifestation_presence[2]  # 实变影
    ggo = abnormal_manifestation_presence[3]  # 磨玻璃影
    
    # 优先级1: OP - 网格影0+蜂窝影0+实变影1+磨玻璃影1/0
    if (reticulation == 0 and 
        honeycombing == 0 and 
        consolidation == 1):
        return "OP"
    
    # 优先级2: UIP - 修正逻辑：(网格影1 OR 蜂窝影1)+磨玻璃影0+实变影0；外周；胸膜下
    if ((reticulation == 1 or honeycombing == 1) and 
        ggo == 0 and 
        consolidation == 0 and
        overall_axial_distribution == "peripheral" and
        overall_manifestation == "subpleural"):
        return "UIP"
    
    # 优先级3: NSIP - 放宽条件：网格影1+蜂窝影0+磨玻璃影1/0+实变影0；外周或弥漫散在；胸膜下省略或胸膜下
    if (reticulation == 1 and 
        honeycombing == 0 and 
        consolidation == 0 and
        overall_axial_distribution in ["peripheral", "diffuse_scattered"] and
        overall_manifestation in ["subpleural_omitted", "subpleural"]):
        return "NSIP"
    
    # 优先级4: HP - 网格影1/0+蜂窝影1/0+磨玻璃影1/0+实变影0；非外周
    if (consolidation == 0 and 
        overall_axial_distribution != "peripheral"):
        return "HP"
    
    # 其他情况
    return "不可分类"

def predict_disease_pattern_original(abnormal_manifestation_presence: List[int], 
                                   overall_axial_distribution: str, 
                                   overall_manifestation: str) -> str:
    """
    原始版疾病模式预测函数（用于对比）
    """
    honeycombing = abnormal_manifestation_presence[0]  # 蜂窝影
    reticulation = abnormal_manifestation_presence[1]  # 网格影
    consolidation = abnormal_manifestation_presence[2]  # 实变影
    ggo = abnormal_manifestation_presence[3]  # 磨玻璃影
    
    # UIP/可能UIP: 网格影1+蜂窝影1/0+磨玻璃影0+实变影0；外周；胸膜下
    if (reticulation == 1 and 
        ggo == 0 and 
        consolidation == 0 and
        overall_axial_distribution == "peripheral" and
        overall_manifestation == "subpleural"):
        return "UIP"
    
    # NSIP: 网格影1+蜂窝影0+磨玻璃影1/0+实变影0；外周或弥漫散在；胸膜下省略
    if (reticulation == 1 and 
        honeycombing == 0 and 
        consolidation == 0 and
        overall_axial_distribution in ["peripheral", "diffuse_scattered"] and
        overall_manifestation == "subpleural_omitted"):
        return "NSIP"
    
    # OP: 网格影0+蜂窝影0+实变影1+磨玻璃影1/0
    if (reticulation == 0 and 
        honeycombing == 0 and 
        consolidation == 1):
        return "OP"
    
    # HP: 网格影1/0+蜂窝影1/0+磨玻璃影1/0+实变影0；非外周
    if (consolidation == 0 and 
        overall_axial_distribution != "peripheral"):
        return "HP"
    
    return "不可分类"

def validate_annotations_comparison(data: List[Dict]) -> Tuple[Dict, Dict]:
    """
    对比原始和改进版验证结果
    
    Returns:
        (原始结果统计, 改进结果统计)
    """
    original_stats = {"total": len(data), "correct": 0, "incorrect": 0, "accuracy": 0.0, "pattern_stats": {}}
    improved_stats = {"total": len(data), "correct": 0, "incorrect": 0, "accuracy": 0.0, "pattern_stats": {}}
    
    comparison_results = []
    
    for item in data:
        filename = item["filename"]
        actual_pattern = item["disease_pattern"]
        
        # 特殊处理：AHP应该对应HP
        actual_pattern_normalized = "HP" if actual_pattern == "AHP" else actual_pattern
        
        # 原始预测
        original_predicted = predict_disease_pattern_original(
            item["abnormal_manifestation_presence"],
            item["overall_axial_distribution"],
            item["overall_manifestation"]
        )
        
        # 改进预测
        improved_predicted = predict_disease_pattern_improved(
            item["abnormal_manifestation_presence"],
            item["overall_axial_distribution"],
            item["overall_manifestation"]
        )
        
        original_correct = original_predicted == actual_pattern_normalized
        improved_correct = improved_predicted == actual_pattern_normalized
        
        # 更新统计
        if original_correct:
            original_stats["correct"] += 1
        else:
            original_stats["incorrect"] += 1
            
        if improved_correct:
            improved_stats["correct"] += 1
        else:
            improved_stats["incorrect"] += 1
        
        # 模式统计
        for stats in [original_stats, improved_stats]:
            if actual_pattern not in stats["pattern_stats"]:
                stats["pattern_stats"][actual_pattern] = {"total": 0, "correct": 0, "accuracy": 0.0}
            stats["pattern_stats"][actual_pattern]["total"] += 1
        
        if original_correct:
            original_stats["pattern_stats"][actual_pattern]["correct"] += 1
        if improved_correct:
            improved_stats["pattern_stats"][actual_pattern]["correct"] += 1
        
        # 记录对比结果
        comparison_results.append({
            "filename": filename,
            "actual_pattern": actual_pattern,
            "original_predicted": original_predicted,
            "improved_predicted": improved_predicted,
            "original_correct": original_correct,
            "improved_correct": improved_correct,
            "improvement": improved_correct and not original_correct,
            "regression": original_correct and not improved_correct,
            "features": {
                "abnormal_manifestation_presence": item["abnormal_manifestation_presence"],
                "overall_axial_distribution": item["overall_axial_distribution"],
                "overall_manifestation": item["overall_manifestation"]
            }
        })
    
    # 计算准确率
    for stats in [original_stats, improved_stats]:
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for pattern in stats["pattern_stats"]:
            pattern_stats = stats["pattern_stats"][pattern]
            pattern_stats["accuracy"] = pattern_stats["correct"] / pattern_stats["total"] if pattern_stats["total"] > 0 else 0.0
    
    return original_stats, improved_stats, comparison_results

def print_comparison_report(original_stats: Dict, improved_stats: Dict, comparison_results: List[Dict]):
    """打印对比报告"""
    print("=" * 80)
    print("原始 vs 改进版验证结果对比报告")
    print("=" * 80)
    
    print(f"\n总体准确率对比:")
    print(f"  原始版本: {original_stats['correct']}/{original_stats['total']} ({original_stats['accuracy']:.2%})")
    print(f"  改进版本: {improved_stats['correct']}/{improved_stats['total']} ({improved_stats['accuracy']:.2%})")
    print(f"  提升幅度: {improved_stats['accuracy'] - original_stats['accuracy']:.2%}")
    
    print(f"\n各疾病模式准确率对比:")
    print(f"{'模式':<8} {'原始准确率':<12} {'改进准确率':<12} {'提升幅度':<10}")
    print("-" * 50)
    
    for pattern in original_stats["pattern_stats"]:
        orig_acc = original_stats["pattern_stats"][pattern]["accuracy"]
        impr_acc = improved_stats["pattern_stats"][pattern]["accuracy"]
        improvement = impr_acc - orig_acc
        print(f"{pattern:<8} {orig_acc:<12.2%} {impr_acc:<12.2%} {improvement:<10.2%}")
    
    # 分析改进和退步的案例
    improvements = [r for r in comparison_results if r["improvement"]]
    regressions = [r for r in comparison_results if r["regression"]]
    
    print(f"\n改进案例分析 (共{len(improvements)}个):")
    if improvements:
        for i, case in enumerate(improvements[:5], 1):  # 显示前5个
            print(f"  {i}. {case['filename']} ({case['actual_pattern']})")
            print(f"     原始: {case['original_predicted']} -> 改进: {case['improved_predicted']}")
            print(f"     特征: {case['features']['abnormal_manifestation_presence']}, "
                  f"{case['features']['overall_axial_distribution']}, "
                  f"{case['features']['overall_manifestation']}")
        if len(improvements) > 5:
            print(f"     ... 还有{len(improvements) - 5}个改进案例")
    
    print(f"\n退步案例分析 (共{len(regressions)}个):")
    if regressions:
        for i, case in enumerate(regressions[:3], 1):  # 显示前3个
            print(f"  {i}. {case['filename']} ({case['actual_pattern']})")
            print(f"     原始: {case['original_predicted']} -> 改进: {case['improved_predicted']}")
            print(f"     特征: {case['features']['abnormal_manifestation_presence']}, "
                  f"{case['features']['overall_axial_distribution']}, "
                  f"{case['features']['overall_manifestation']}")
    else:
        print("  无退步案例")

def save_comparison_results(original_stats: Dict, improved_stats: Dict, 
                          comparison_results: List[Dict], output_file: str):
    """保存对比结果"""
    output_data = {
        "original_validation_stats": original_stats,
        "improved_validation_stats": improved_stats,
        "comparison_results": comparison_results,
        "summary": {
            "total_samples": len(comparison_results),
            "improvements": len([r for r in comparison_results if r["improvement"]]),
            "regressions": len([r for r in comparison_results if r["regression"]]),
            "accuracy_improvement": improved_stats["accuracy"] - original_stats["accuracy"]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n对比结果已保存到: {output_file}")

def main():
    """主函数"""
    input_file = "/home/huchengpeng/workspace/dataset/annotations/processed/processed_labels.jsonl"
    output_file = "/home/huchengpeng/workspace/dataset/annotations/processed/validation_comparison_results.json"
    
    print("加载数据...")
    data = load_jsonl(input_file)
    
    print("执行对比验证...")
    original_stats, improved_stats, comparison_results = validate_annotations_comparison(data)
    
    print_comparison_report(original_stats, improved_stats, comparison_results)
    save_comparison_results(original_stats, improved_stats, comparison_results, output_file)
    
    print("\n对比验证完成！")

if __name__ == "__main__":
    main()