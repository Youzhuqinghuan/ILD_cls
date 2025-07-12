#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学影像标注验证程序
根据模式判别逻辑验证数据集标注的正确性

模式判别逻辑：
1. UIP/可能UIP: 网格影1+蜂窝影1/0+磨玻璃影0+实变影0；外周；胸膜下
2. NSIP: 网格影1+蜂窝影0+磨玻璃影1/0+实变影0；外周或弥漫散在；胸膜下省略
3. OP: 网格影0+蜂窝影0+实变影1+磨玻璃影1/0
4. HP: 网格影1/0+蜂窝影1/0+磨玻璃影1/0+实变影0；非外周
5. 其他: 不可分类

注：由于缺少上/下肺占比变量，此处仅通过abnormal_manifestation_presence、
overall_axial_distribution、overall_manifestation三个变量进行验证
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

def predict_disease_pattern(abnormal_manifestation_presence: List[int], 
                          overall_axial_distribution: str, 
                          overall_manifestation: str) -> str:
    """
    根据模式判别逻辑预测疾病模式
    
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
    
    # 其他情况
    return "不可分类"

def validate_annotations(data: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    验证标注数据
    
    Returns:
        (验证结果列表, 统计信息)
    """
    results = []
    stats = {
        "total": len(data),
        "correct": 0,
        "incorrect": 0,
        "accuracy": 0.0,
        "pattern_stats": {},
        "error_details": []
    }
    
    for item in data:
        filename = item["filename"]
        actual_pattern = item["disease_pattern"]
        predicted_pattern = predict_disease_pattern(
            item["abnormal_manifestation_presence"],
            item["overall_axial_distribution"],
            item["overall_manifestation"]
        )
        
        # 特殊处理：AHP应该对应HP
        if actual_pattern == "AHP":
            actual_pattern_normalized = "HP"
        else:
            actual_pattern_normalized = actual_pattern
        
        is_correct = predicted_pattern == actual_pattern_normalized
        
        result = {
            "filename": filename,
            "actual_pattern": item["disease_pattern"],
            "predicted_pattern": predicted_pattern,
            "is_correct": is_correct,
            "abnormal_manifestation_presence": item["abnormal_manifestation_presence"],
            "overall_axial_distribution": item["overall_axial_distribution"],
            "overall_manifestation": item["overall_manifestation"]
        }
        
        results.append(result)
        
        # 更新统计信息
        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
            stats["error_details"].append({
                "filename": filename,
                "actual": item["disease_pattern"],
                "predicted": predicted_pattern,
                "features": {
                    "abnormal_manifestation_presence": item["abnormal_manifestation_presence"],
                    "overall_axial_distribution": item["overall_axial_distribution"],
                    "overall_manifestation": item["overall_manifestation"]
                }
            })
        
        # 统计各模式的情况
        actual_key = item["disease_pattern"]
        if actual_key not in stats["pattern_stats"]:
            stats["pattern_stats"][actual_key] = {"total": 0, "correct": 0, "accuracy": 0.0}
        
        stats["pattern_stats"][actual_key]["total"] += 1
        if is_correct:
            stats["pattern_stats"][actual_key]["correct"] += 1
    
    # 计算准确率
    stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    for pattern in stats["pattern_stats"]:
        pattern_stats = stats["pattern_stats"][pattern]
        pattern_stats["accuracy"] = pattern_stats["correct"] / pattern_stats["total"] if pattern_stats["total"] > 0 else 0.0
    
    return results, stats

def print_validation_report(stats: Dict, results: List[Dict]):
    """打印验证报告"""
    print("=" * 60)
    print("医学影像标注验证报告")
    print("=" * 60)
    
    print(f"\n总体统计:")
    print(f"  总样本数: {stats['total']}")
    print(f"  正确预测: {stats['correct']}")
    print(f"  错误预测: {stats['incorrect']}")
    print(f"  总体准确率: {stats['accuracy']:.2%}")
    
    print(f"\n各疾病模式准确率:")
    for pattern, pattern_stats in stats["pattern_stats"].items():
        print(f"  {pattern}: {pattern_stats['correct']}/{pattern_stats['total']} ({pattern_stats['accuracy']:.2%})")
    
    if stats["error_details"]:
        print(f"\n错误预测详情 (共{len(stats['error_details'])}个):")
        for i, error in enumerate(stats["error_details"][:10], 1):  # 只显示前10个错误
            print(f"  {i}. {error['filename']}")
            print(f"     实际: {error['actual']} -> 预测: {error['predicted']}")
            print(f"     特征: 异常表现{error['features']['abnormal_manifestation_presence']}, "
                  f"轴向分布={error['features']['overall_axial_distribution']}, "
                  f"表现形式={error['features']['overall_manifestation']}")
        
        if len(stats["error_details"]) > 10:
            print(f"     ... 还有{len(stats['error_details']) - 10}个错误未显示")

def save_validation_results(results: List[Dict], stats: Dict, output_file: str):
    """保存验证结果"""
    output_data = {
        "validation_stats": stats,
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n验证结果已保存到: {output_file}")

def main():
    """主函数"""
    input_file = "/home/huchengpeng/workspace/dataset/annotations/processed/processed_labels.jsonl"
    output_file = "/home/huchengpeng/workspace/dataset/annotations/processed/validation_results.json"
    
    print("加载数据...")
    data = load_jsonl(input_file)
    
    print("验证标注...")
    results, stats = validate_annotations(data)
    
    print_validation_report(stats, results)
    save_validation_results(results, stats, output_file)
    
    print("\n验证完成！")

if __name__ == "__main__":
    main()