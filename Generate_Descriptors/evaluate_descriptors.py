#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放射学描述符评估程序

该程序用于比较标注数据和算法预测结果中的放射学描述符，计算预测准确率。
比较的描述符包括：
1. 轴向分布 (overall_axial_distribution vs axial_distribution)
2. 整体征象 (overall_manifestation vs manifestation)

作者: AI Assistant
日期: 2024
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载JSONL文件
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        包含所有记录的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def normalize_case_name(name: str) -> str:
    """
    标准化病例名称，处理命名差异
    
    Args:
        name: 原始病例名称
        
    Returns:
        标准化后的病例名称
    """
    # 将"NSIP 1"格式转换为"NSIP1"格式
    return name.replace(" ", "")

def create_case_mapping(annotations: List[Dict], predictions: List[Dict]) -> Dict[str, Tuple[Dict, Dict]]:
    """
    创建病例映射，匹配标注数据和预测结果
    
    Args:
        annotations: 标注数据列表
        predictions: 预测结果列表
        
    Returns:
        病例映射字典，键为标准化病例名称，值为(标注数据, 预测数据)元组
    """
    # 创建标注数据映射
    ann_map = {}
    for ann in annotations:
        case_name = normalize_case_name(ann['filename'])
        ann_map[case_name] = ann
    
    # 创建预测数据映射
    pred_map = {}
    for pred in predictions:
        case_name = normalize_case_name(pred['case_name'])
        pred_map[case_name] = pred
    
    # 找到共同的病例
    common_cases = set(ann_map.keys()) & set(pred_map.keys())
    
    # 创建匹配映射
    case_mapping = {}
    for case_name in common_cases:
        case_mapping[case_name] = (ann_map[case_name], pred_map[case_name])
    
    return case_mapping

def evaluate_descriptors(case_mapping: Dict[str, Tuple[Dict, Dict]]) -> Dict[str, Dict]:
    """
    评估描述符预测准确率
    
    Args:
        case_mapping: 病例映射字典
        
    Returns:
        评估结果字典
    """
    results = {
        'axial_distribution': {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0,
            'details': []
        },
        'manifestation': {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0,
            'details': []
        }
    }
    
    for case_name, (annotation, prediction) in case_mapping.items():
        # 评估轴向分布
        ann_axial = annotation['overall_axial_distribution']
        pred_axial = prediction['axial_distribution']
        axial_correct = ann_axial == pred_axial
        
        results['axial_distribution']['total'] += 1
        if axial_correct:
            results['axial_distribution']['correct'] += 1
        
        results['axial_distribution']['details'].append({
            'case_name': case_name,
            'annotation': ann_axial,
            'prediction': pred_axial,
            'correct': axial_correct
        })
        
        # 评估整体征象
        ann_manifestation = annotation['overall_manifestation']
        pred_manifestation = prediction['manifestation']
        manifestation_correct = ann_manifestation == pred_manifestation
        
        results['manifestation']['total'] += 1
        if manifestation_correct:
            results['manifestation']['correct'] += 1
        
        results['manifestation']['details'].append({
            'case_name': case_name,
            'annotation': ann_manifestation,
            'prediction': pred_manifestation,
            'correct': manifestation_correct
        })
    
    # 计算准确率
    if results['axial_distribution']['total'] > 0:
        results['axial_distribution']['accuracy'] = results['axial_distribution']['correct'] / results['axial_distribution']['total']
    
    if results['manifestation']['total'] > 0:
        results['manifestation']['accuracy'] = results['manifestation']['correct'] / results['manifestation']['total']
    
    return results

def print_evaluation_summary(results: Dict[str, Dict]):
    """
    打印评估结果摘要
    
    Args:
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("放射学描述符评估结果")
    print("="*60)
    
    # 轴向分布评估结果
    axial_results = results['axial_distribution']
    print(f"\n1. 轴向分布 (Axial Distribution):")
    print(f"   正确预测: {axial_results['correct']}/{axial_results['total']}")
    print(f"   准确率: {axial_results['accuracy']:.2%}")
    
    # 整体征象评估结果
    manifestation_results = results['manifestation']
    print(f"\n2. 整体征象 (Manifestation):")
    print(f"   正确预测: {manifestation_results['correct']}/{manifestation_results['total']}")
    print(f"   准确率: {manifestation_results['accuracy']:.2%}")
    
    # 总体准确率
    total_correct = axial_results['correct'] + manifestation_results['correct']
    total_predictions = axial_results['total'] + manifestation_results['total']
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    print(f"\n3. 总体准确率:")
    print(f"   正确预测: {total_correct}/{total_predictions}")
    print(f"   准确率: {overall_accuracy:.2%}")
    print("="*60)

def print_detailed_results(results: Dict[str, Dict], show_errors_only: bool = False):
    """
    打印详细评估结果
    
    Args:
        results: 评估结果字典
        show_errors_only: 是否只显示错误预测
    """
    print("\n详细评估结果:")
    print("-"*60)
    
    # 轴向分布详细结果
    print("\n轴向分布 (Axial Distribution):")
    for detail in results['axial_distribution']['details']:
        if show_errors_only and detail['correct']:
            continue
        status = "✓" if detail['correct'] else "✗"
        print(f"  {status} {detail['case_name']}: {detail['annotation']} -> {detail['prediction']}")
    
    # 整体征象详细结果
    print("\n整体征象 (Manifestation):")
    for detail in results['manifestation']['details']:
        if show_errors_only and detail['correct']:
            continue
        status = "✓" if detail['correct'] else "✗"
        print(f"  {status} {detail['case_name']}: {detail['annotation']} -> {detail['prediction']}")

def save_evaluation_results(results: Dict[str, Dict], output_path: str):
    """
    保存评估结果到JSON文件
    
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='评估放射学描述符预测准确率')
    parser.add_argument('--annotations', 
                       default='/home/huchengpeng/workspace/dataset/annotations/processed/processed_labels.jsonl',
                       help='标注数据文件路径')
    parser.add_argument('--predictions', 
                       default='/home/huchengpeng/workspace/Generate_Descriptors/radiological_descriptors_all.jsonl',
                       help='预测结果文件路径')
    parser.add_argument('--output', 
                       default='/home/huchengpeng/workspace/Generate_Descriptors/evaluation_results.json',
                       help='评估结果输出文件路径')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细评估结果')
    parser.add_argument('--errors-only', action='store_true',
                       help='只显示错误预测（需要与--detailed一起使用）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.annotations).exists():
        print(f"错误: 标注数据文件不存在: {args.annotations}")
        return
    
    if not Path(args.predictions).exists():
        print(f"错误: 预测结果文件不存在: {args.predictions}")
        return
    
    print(f"加载标注数据: {args.annotations}")
    annotations = load_jsonl(args.annotations)
    print(f"加载了 {len(annotations)} 个标注样本")
    
    print(f"加载预测结果: {args.predictions}")
    predictions = load_jsonl(args.predictions)
    print(f"加载了 {len(predictions)} 个预测样本")
    
    # 创建病例映射
    case_mapping = create_case_mapping(annotations, predictions)
    print(f"匹配到 {len(case_mapping)} 个共同病例")
    
    if len(case_mapping) == 0:
        print("警告: 没有找到匹配的病例，请检查文件格式和病例命名")
        return
    
    # 评估描述符
    results = evaluate_descriptors(case_mapping)
    
    # 打印评估摘要
    print_evaluation_summary(results)
    
    # 打印详细结果（如果需要）
    if args.detailed:
        print_detailed_results(results, args.errors_only)
    
    # 保存评估结果
    save_evaluation_results(results, args.output)

if __name__ == '__main__':
    main()