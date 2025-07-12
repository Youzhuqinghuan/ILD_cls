"""
Pattern classifier for ILD Analysis
Integrates four variables to classify ILD patterns: UIP, NSIP, OP, Other, Normal
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ILDPatternClassifier:
    """Classifies ILD patterns based on four extracted variables"""
    
    def __init__(self):
        """Initialize ILD pattern classifier with rule-based logic"""
        self.patterns = ["UIP", "NSIP", "OP", "Other", "Normal"]
        
        # Classification thresholds
        self.thresholds = {
            "minimal_lesion_threshold": 0.02,  # 2% lung involvement
            "significant_lesion_threshold": 0.10,  # 10% lung involvement
            "predominance_threshold": 0.65,  # 65% for upper/lower predominance
            "peripheral_threshold": 0.70,  # 70% for peripheral classification
            "subpleural_threshold": 0.30  # 30% for significant subpleural involvement
        }
    
    def classify_pattern(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify ILD pattern based on extracted features
        
        Args:
            features: Dictionary containing results from all four feature extractors
            
        Returns:
            Dictionary with pattern classification results
        """
        classification = {
            "predicted_pattern": "Normal",
            "confidence": 0.0,
            "scores": {pattern: 0.0 for pattern in self.patterns},
            "reasoning": [],
            "differential_diagnosis": [],
            "certainty_level": "low"
        }
        
        try:
            # Extract key features
            lesion_features = features.get("lesion_type", {})
            lung_distribution = features.get("lung_distribution", {})
            axial_distribution = features.get("axial_distribution", {})
            subpleural_features = features.get("subpleural_features", {})
            
            # Check if case is normal (minimal lesions)
            if self._is_normal_case(lesion_features, lung_distribution):
                classification["predicted_pattern"] = "Normal"
                classification["confidence"] = 0.9
                classification["scores"]["Normal"] = 0.9
                classification["reasoning"].append("Minimal or no lesions detected")
                classification["certainty_level"] = "high"
                return classification
            
            # Calculate scores for each pattern
            uip_score = self._calculate_uip_score(
                lesion_features, lung_distribution, axial_distribution, subpleural_features
            )
            nsip_score = self._calculate_nsip_score(
                lesion_features, lung_distribution, axial_distribution, subpleural_features
            )
            op_score = self._calculate_op_score(
                lesion_features, lung_distribution, axial_distribution, subpleural_features
            )
            
            # Store scores
            classification["scores"]["UIP"] = uip_score["score"]
            classification["scores"]["NSIP"] = nsip_score["score"]
            classification["scores"]["OP"] = op_score["score"]
            classification["scores"]["Other"] = max(0.0, 1.0 - max(uip_score["score"], nsip_score["score"], op_score["score"]))
            
            # Determine predicted pattern
            max_score = max(classification["scores"].values())
            predicted_pattern = max(classification["scores"], key=classification["scores"].get)
            
            classification["predicted_pattern"] = predicted_pattern
            classification["confidence"] = max_score
            
            # Combine reasoning from all patterns
            classification["reasoning"].extend(uip_score["reasoning"])
            classification["reasoning"].extend(nsip_score["reasoning"])
            classification["reasoning"].extend(op_score["reasoning"])
            
            # Determine certainty level
            classification["certainty_level"] = self._determine_certainty_level(classification["scores"])
            
            # Generate differential diagnosis
            classification["differential_diagnosis"] = self._generate_differential_diagnosis(classification["scores"])
            
            logger.info(f"Classification result: {predicted_pattern} (confidence: {max_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error in pattern classification: {str(e)}")
            classification["error"] = str(e)
        
        return classification
    
    def _is_normal_case(self, lesion_features: Dict, lung_distribution: Dict) -> bool:
        """Check if case should be classified as normal"""
        try:
            # Check lesion coverage
            lesion_types = lesion_features.get("lesion_types_present", [])
            if not lesion_types or lesion_types == ["background"]:
                return True
            
            # Check total lesion ratio
            total_ratio = lung_distribution.get("total_lesion_volume", 0) / max(
                lung_distribution.get("total_lung_volume", 1), 1
            )
            
            return total_ratio < self.thresholds["minimal_lesion_threshold"]
            
        except Exception as e:
            logger.error(f"Error checking normal case: {str(e)}")
            return False
    
    def _calculate_uip_score(self, lesion_features: Dict, lung_distribution: Dict,
                           axial_distribution: Dict, subpleural_features: Dict) -> Dict[str, Any]:
        """Calculate UIP pattern score"""
        score = 0.0
        reasoning = []
        
        try:
            # Honeycomb pattern (strong UIP indicator)
            lesion_types = lesion_features.get("lesion_types_present", [])
            dominant_lesion = lesion_features.get("dominant_lesion")
            
            if "honeycomb" in lesion_types:
                score += 0.4
                reasoning.append("Honeycomb pattern present (strong UIP indicator)")
                
                if dominant_lesion == "honeycomb":
                    score += 0.2
                    reasoning.append("Honeycomb is dominant lesion")
            
            # Lower lobe predominance
            distribution_pattern = lung_distribution.get("distribution_pattern", "unknown")
            if distribution_pattern == "lower_predominant":
                score += 0.2
                reasoning.append("Lower lobe predominance")
            elif distribution_pattern == "upper_predominant":
                score -= 0.1
                reasoning.append("Upper lobe predominance (atypical for UIP)")
            
            # Peripheral distribution
            axial_pattern = axial_distribution.get("axial_distribution", "unknown")
            if axial_pattern == "peripheral":
                score += 0.15
                reasoning.append("Peripheral distribution")
            elif axial_pattern == "central":
                score -= 0.1
                reasoning.append("Central distribution (atypical for UIP)")
            
            # Subpleural involvement
            subpleural_involvement = subpleural_features.get("subpleural_involvement", False)
            if subpleural_involvement:
                score += 0.15
                reasoning.append("Subpleural involvement")
                
                # Extensive subpleural involvement
                subpleural_ratio = subpleural_features.get("subpleural_lesion_ratio", 0)
                if subpleural_ratio > self.thresholds["subpleural_threshold"]:
                    score += 0.1
                    reasoning.append("Extensive subpleural involvement")
            else:
                score -= 0.05
                reasoning.append("No subpleural involvement (atypical for UIP)")
            
            # Reticulation pattern (supportive)
            if "reticulation" in lesion_types:
                score += 0.05
                reasoning.append("Reticulation pattern present")
            
            # Ground glass (should be minimal in UIP)
            if dominant_lesion == "ground_glass_opacity":
                score -= 0.1
                reasoning.append("Extensive ground glass (atypical for UIP)")
            
        except Exception as e:
            logger.error(f"Error calculating UIP score: {str(e)}")
            reasoning.append(f"Error in UIP scoring: {str(e)}")
        
        return {"score": max(0.0, min(1.0, score)), "reasoning": reasoning}
    
    def _calculate_nsip_score(self, lesion_features: Dict, lung_distribution: Dict,
                            axial_distribution: Dict, subpleural_features: Dict) -> Dict[str, Any]:
        """Calculate NSIP pattern score"""
        score = 0.0
        reasoning = []
        
        try:
            lesion_types = lesion_features.get("lesion_types_present", [])
            dominant_lesion = lesion_features.get("dominant_lesion")
            
            # Ground glass opacity (common in NSIP)
            if "ground_glass_opacity" in lesion_types:
                score += 0.2
                reasoning.append("Ground glass opacity present")
                
                if dominant_lesion == "ground_glass_opacity":
                    score += 0.2
                    reasoning.append("Ground glass is dominant")
            
            # Reticulation (fibrotic NSIP)
            if "reticulation" in lesion_types:
                score += 0.15
                reasoning.append("Reticulation pattern present")
            
            # Absence of honeycomb (typical for NSIP)
            if "honeycomb" not in lesion_types:
                score += 0.15
                reasoning.append("No honeycomb pattern (typical for NSIP)")
            else:
                score -= 0.2
                reasoning.append("Honeycomb present (atypical for NSIP)")
            
            # Distribution patterns
            distribution_pattern = lung_distribution.get("distribution_pattern", "unknown")
            if distribution_pattern in ["diffuse", "upper_predominant"]:
                score += 0.1
                reasoning.append(f"{distribution_pattern} distribution")
            elif distribution_pattern == "lower_predominant":
                score += 0.05  # Can occur but less typical
                reasoning.append("Lower predominance (less typical for NSIP)")
            
            # Axial distribution (can be peripheral or diffuse)
            axial_pattern = axial_distribution.get("axial_distribution", "unknown")
            if axial_pattern in ["peripheral", "diffuse_scattered"]:
                score += 0.1
                reasoning.append(f"{axial_pattern} axial distribution")
            
            # Subpleural sparing (classic NSIP feature)
            subpleural_involvement = subpleural_features.get("subpleural_involvement", False)
            if not subpleural_involvement:
                score += 0.2
                reasoning.append("Subpleural sparing (classic NSIP feature)")
            else:
                # Minimal subpleural involvement can occur
                subpleural_ratio = subpleural_features.get("subpleural_lesion_ratio", 0)
                if subpleural_ratio < 0.1:  # <10%
                    score += 0.05
                    reasoning.append("Minimal subpleural involvement")
                else:
                    score -= 0.1
                    reasoning.append("Significant subpleural involvement (atypical for NSIP)")
            
        except Exception as e:
            logger.error(f"Error calculating NSIP score: {str(e)}")
            reasoning.append(f"Error in NSIP scoring: {str(e)}")
        
        return {"score": max(0.0, min(1.0, score)), "reasoning": reasoning}
    
    def _calculate_op_score(self, lesion_features: Dict, lung_distribution: Dict,
                          axial_distribution: Dict, subpleural_features: Dict) -> Dict[str, Any]:
        """Calculate Organizing Pneumonia (OP) pattern score"""
        score = 0.0
        reasoning = []
        
        try:
            lesion_types = lesion_features.get("lesion_types_present", [])
            dominant_lesion = lesion_features.get("dominant_lesion")
            
            # Consolidation (hallmark of OP)
            if "consolidation" in lesion_types:
                score += 0.3
                reasoning.append("Consolidation present")
                
                if dominant_lesion == "consolidation":
                    score += 0.2
                    reasoning.append("Consolidation is dominant")
            
            # Ground glass with consolidation
            if "ground_glass_opacity" in lesion_types and "consolidation" in lesion_types:
                score += 0.2
                reasoning.append("Mixed ground glass and consolidation")
            elif dominant_lesion == "ground_glass_opacity":
                score += 0.1
                reasoning.append("Ground glass dominant (possible early OP)")
            
            # Absence of honeycomb and reticulation (typical for OP)
            if "honeycomb" not in lesion_types and "reticulation" not in lesion_types:
                score += 0.1
                reasoning.append("No fibrotic patterns (typical for OP)")
            
            # Distribution patterns (OP can be variable)
            distribution_pattern = lung_distribution.get("distribution_pattern", "unknown")
            if distribution_pattern in ["diffuse", "lower_predominant"]:
                score += 0.05
                reasoning.append(f"{distribution_pattern} distribution")
            
            # Peripheral or subpleural distribution
            axial_pattern = axial_distribution.get("axial_distribution", "unknown")
            if axial_pattern == "peripheral":
                score += 0.1
                reasoning.append("Peripheral distribution")
            
            # Subpleural involvement (can occur in OP)
            subpleural_involvement = subpleural_features.get("subpleural_involvement", False)
            if subpleural_involvement:
                score += 0.05
                reasoning.append("Subpleural involvement")
            
        except Exception as e:
            logger.error(f"Error calculating OP score: {str(e)}")
            reasoning.append(f"Error in OP scoring: {str(e)}")
        
        return {"score": max(0.0, min(1.0, score)), "reasoning": reasoning}
    
    def _determine_certainty_level(self, scores: Dict[str, float]) -> str:
        """Determine certainty level based on score distribution"""
        try:
            sorted_scores = sorted(scores.values(), reverse=True)
            max_score = sorted_scores[0]
            second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
            
            if max_score >= 0.8:
                return "high"
            elif max_score >= 0.6 and (max_score - second_score) >= 0.2:
                return "moderate"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Error determining certainty level: {str(e)}")
            return "low"
    
    def _generate_differential_diagnosis(self, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate differential diagnosis list"""
        try:
            # Sort patterns by score
            sorted_patterns = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            differential = []
            for pattern, score in sorted_patterns[:3]:  # Top 3
                if score > 0.1:  # Only include patterns with meaningful scores
                    differential.append({
                        "pattern": pattern,
                        "probability": score,
                        "likelihood": self._score_to_likelihood(score)
                    })
            
            return differential
            
        except Exception as e:
            logger.error(f"Error generating differential diagnosis: {str(e)}")
            return []
    
    def _score_to_likelihood(self, score: float) -> str:
        """Convert score to likelihood description"""
        if score >= 0.8:
            return "Very likely"
        elif score >= 0.6:
            return "Likely"
        elif score >= 0.4:
            return "Possible"
        elif score >= 0.2:
            return "Unlikely"
        else:
            return "Very unlikely"
    
    def get_classification_summary(self, classification: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable classification summary"""
        summary = {
            "pattern": classification.get("predicted_pattern", "Unknown"),
            "confidence": f"{classification.get('confidence', 0) * 100:.1f}%",
            "certainty": classification.get("certainty_level", "low").title(),
            "top_differential": "None"
        }
        
        try:
            differential = classification.get("differential_diagnosis", [])
            if len(differential) > 1:
                second_best = differential[1]
                summary["top_differential"] = (
                    f"{second_best['pattern']} ({second_best['probability'] * 100:.1f}%)"
                )
            
        except Exception as e:
            logger.error(f"Error generating classification summary: {str(e)}")
            summary["error"] = str(e)
        
        return summary