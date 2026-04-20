# modules/ai_engine.py
"""
Moteur d'IA avec capacité d'apprentissage et d'analyse contextuelle
Version améliorée avec génération de phrases naturelles
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


class AIAnalyzer:
    """
    Moteur d'IA avancé avec capacité d'apprentissage et de mémoire
    """
    
    def __init__(self, memory_path="ai_memory.pkl"):
        self.memory_path = memory_path
        self.memory = self._load_memory()
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
        # Plafonds réglementaires
        self.plafonds = {
            "BL": 90,
            "RG": 90,
            "RS": 100
        }
    
    def _load_memory(self) -> Dict:
        """Charge la mémoire de l'IA depuis le disque"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return self._init_memory()
        return self._init_memory()
    
    def _init_memory(self) -> Dict:
        """Initialise une nouvelle mémoire"""
        return {
            'analyses': [],
            'patterns': {},
            'thresholds': {},
            'feedback': [],
            'learning_history': []
        }
    
    def _save_memory(self):
        """Sauvegarde la mémoire de l'IA"""
        with open(self.memory_path, 'wb') as f:
            pickle.dump(self.memory, f)
    
    def _detect_patterns(self, data: pd.DataFrame, column: str) -> Dict:
        """Détecte des patterns dans les données"""
        patterns = {}
        
        if column in data.columns and len(data) > 3:
            # Préparation des données
            if 'annee' in data.columns:
                data_sorted = data.sort_values('annee')
                y = data_sorted[column].values
                x = data_sorted['annee'].values
            else:
                y = data[column].values
                x = np.arange(len(y))
            
            if len(y) > 2 and len(np.unique(y)) > 1:
                # Tendance (pente par an)
                coeffs = np.polyfit(x, y, 1)
                patterns['trend'] = coeffs[0]
                patterns['trend_pct'] = (coeffs[0] / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                
                # Volatilité
                patterns['volatility'] = np.std(y)
                patterns['cv'] = (np.std(y) / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                
                # Classification de la stabilité
                if patterns['cv'] < 15:
                    patterns['stability'] = 'très stable'
                elif patterns['cv'] < 25:
                    patterns['stability'] = 'stable'
                elif patterns['cv'] < 35:
                    patterns['stability'] = 'modérément variable'
                else:
                    patterns['stability'] = 'très variable'
                
                patterns['min_year'] = x[np.argmin(y)]
                patterns['max_year'] = x[np.argmax(y)]
                patterns['min_value'] = np.min(y)
                patterns['max_value'] = np.max(y)
                patterns['range'] = patterns['max_value'] - patterns['min_value']
        
        return patterns
    
    def _generate_rendement_analysis(self, data: Dict) -> str:
        """Génère une analyse en langage naturel pour les rendements"""
        
        lines = []
        stats = data.get('stats', {})
        patterns = data.get('patterns', {})
        risks = data.get('risks', [])
        recommendations = data.get('recommendations', [])
        comparisons = data.get('comparisons', {})
        
        if not stats:
            return "Données insuffisantes pour l'analyse."
        
        # Calcul des moyennes globales
        means = [s['mean'] for s in stats.values()]
        stds = [s['std'] for s in stats.values()]
        
        global_mean = np.mean(means)
        global_std = np.mean(stds)
        
        # Introduction
        lines.append(f"**Analyse globale des rendements**")
        
        if len(stats) == 1:
            groupe = list(stats.keys())[0]
            lines.append(f"Pour la catégorie **{groupe}**, le rendement moyen observé est de **{global_mean:.1f} hl/ha**.")
        else:
            lines.append(f"L'analyse porte sur {len(stats)} catégories. Le rendement moyen global est de **{global_mean:.1f} hl/ha**, avec une variabilité moyenne de **{global_std:.1f} hl/ha**.")
        
        lines.append("")
        
        # Analyse par groupe
        lines.append(f"**Analyse détaillée par catégorie :**")
        lines.append("")
        
        for groupe, stat in stats.items():
            lines.append(f"**{groupe}** :")
            
            # Position par rapport à la moyenne
            if len(stats) > 1:
                if stat['mean'] > global_mean + 3:
                    lines.append(f"  • Rendement **supérieur** à la moyenne de {stat['mean'] - global_mean:.1f} hl/ha.")
                elif stat['mean'] < global_mean - 3:
                    lines.append(f"  • Rendement **inférieur** à la moyenne de {global_mean - stat['mean']:.1f} hl/ha.")
                else:
                    lines.append(f"  • Rendement **dans la moyenne** globale.")
            
            # Variabilité
            cv = (stat['std'] / stat['mean'] * 100) if stat['mean'] > 0 else 0
            
            if cv > 35:
                lines.append(f"  • **Forte variabilité** interannuelle (écart-type de {stat['std']:.1f} hl/ha, CV={cv:.0f}%). Cette irrégularité complique la planification et peut indiquer une sensibilité aux aléas climatiques.")
            elif cv > 25:
                lines.append(f"  • **Variabilité marquée** (écart-type de {stat['std']:.1f} hl/ha, CV={cv:.0f}%). La production fluctue significativement selon les millésimes.")
            elif cv > 15:
                lines.append(f"  • **Variabilité modérée** (écart-type de {stat['std']:.1f} hl/ha, CV={cv:.0f}%).")
            else:
                lines.append(f"  • **Très bonne stabilité** (écart-type de {stat['std']:.1f} hl/ha, CV={cv:.0f}%).")
            
            # Étendue
            lines.append(f"  • Les rendements s'échelonnent de **{stat['min']:.0f}** à **{stat['max']:.0f} hl/ha**, soit une amplitude de {stat['max'] - stat['min']:.0f} hl/ha.")
            
            # Analyse des minimums
            if stat['min'] < 20:
                lines.append(f"  • Les années de très faible production (<20 hl/ha) évoquent des **épisodes de gel printanier** ou de **sécheresse sévère**.")
            
            # Proximité des plafonds
            plafond = self.plafonds.get(groupe, 90)
            if stat['max'] > plafond:
                lines.append(f"  • Le rendement maximum ({stat['max']:.0f} hl/ha) **dépasse le plafond réglementaire** de {plafond} hl/ha. Ces années présentent un risque de déclassement.")
            elif stat['max'] > plafond * 0.9:
                lines.append(f"  • Le rendement maximum ({stat['max']:.0f} hl/ha) est **proche du plafond** de {plafond} hl/ha.")
            
            # Analyse de la tendance
            if groupe in patterns and 'trend' in patterns[groupe]:
                trend = patterns[groupe]['trend']
                if abs(trend) > 0.3:
                    if trend > 0:
                        lines.append(f"  • **Tendance à la hausse** : +{trend:.2f} hl/ha par an.")
                    else:
                        lines.append(f"  • **Tendance à la baisse** : {trend:.2f} hl/ha par an.")
            
            lines.append("")
        
        # Comparaisons
        if comparisons:
            lines.append(f"**Comparaison entre catégories :**")
            lines.append("")
            lines.append(f"  • La catégorie la plus productive est **{comparisons['best']}** avec {comparisons['best_value']:.1f} hl/ha.")
            lines.append(f"  • La catégorie la moins productive est **{comparisons['worst']}** avec {comparisons['worst_value']:.1f} hl/ha.")
            lines.append(f"  • L'écart est de **{comparisons['gap']:.1f} hl/ha** ({comparisons['gap_percent']:.0f}%).")
            lines.append("")
        
        # Risques
        if risks:
            lines.append(f"**Synthèse des risques identifiés :**")
            lines.append("")
            for risk in risks[:5]:  # Limiter à 5 risques
                lines.append(f"  • {risk}")
            lines.append("")
        
        # Recommandations
        if recommendations:
            lines.append(f"**Recommandations personnalisées :**")
            lines.append("")
            for rec in recommendations[:3]:  # Limiter à 3 recommandations
                lines.append(f"  • {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def analyze_rendement(self, data: pd.DataFrame, mode: str, selections: List[str]) -> Dict[str, Any]:
        """
        Analyse approfondie des rendements
        """
        df_analysis = data.copy()
        
        # Mapping des modes vers les noms de colonnes réels
        mode_to_col = {
            "Couleur": "code_couleur",
            "Departement": "code_departement",  # Sans accent
            "Zone": "Zone"
        }
        
        col_name = mode_to_col.get(mode, mode)
        
        stats = {}
        patterns = {}
        recommendations = []
        risks = []
        comparisons = {}
        
        for selection in selections:
            subset = df_analysis[df_analysis[col_name] == selection]
            
            if not subset.empty:
                rendements = subset['rendement'].dropna()
                
                if len(rendements) > 0:
                    stats[selection] = {
                        'mean': rendements.mean(),
                        'median': rendements.median(),
                        'std': rendements.std(),
                        'min': rendements.min(),
                        'max': rendements.max(),
                        'q1': rendements.quantile(0.25),
                        'q3': rendements.quantile(0.75),
                        'n': len(rendements),
                        'n_years': len(subset['annee'].unique())
                    }
                    
                    # Patterns
                    time_data = subset.groupby('annee')['rendement'].mean().reset_index()
                    patterns[selection] = self._detect_patterns(time_data, 'rendement')
                    
                    # Risques
                    cv = (stats[selection]['std'] / stats[selection]['mean'] * 100) if stats[selection]['mean'] > 0 else 0
                    
                    if cv > 35:
                        risks.append(f"**{selection}** : Très forte variabilité (CV={cv:.0f}%)")
                    
                    if stats[selection]['min'] < 20:
                        risks.append(f"**{selection}** : Années avec rendements <20 hl/ha")
                    
                    plafond = self.plafonds.get(selection, 90)
                    if stats[selection]['max'] > plafond:
                        risks.append(f"**{selection}** : Dépassement du plafond de {plafond} hl/ha")
                    
                    # Tendances
                    if selection in patterns and 'trend' in patterns[selection]:
                        trend = patterns[selection]['trend']
                        if abs(trend) > 0.5:
                            if trend > 0:
                                recommendations.append(f"**{selection}** : Tendance haussière de {trend:.2f} hl/ha/an")
                            else:
                                recommendations.append(f"**{selection}** : Tendance baissière de {abs(trend):.2f} hl/ha/an")
        
        # Comparaisons
        if len(selections) > 1:
            means = {k: v['mean'] for k, v in stats.items()}
            best = max(means, key=means.get)
            worst = min(means, key=means.get)
            
            comparisons = {
                'best': best,
                'worst': worst,
                'best_value': means[best],
                'worst_value': means[worst],
                'gap': means[best] - means[worst],
                'gap_percent': ((means[best] - means[worst]) / means[worst] * 100) if means[worst] > 0 else 0
            }
        
        # Génération analyse
        analysis_data = {
            'stats': stats,
            'patterns': patterns,
            'risks': risks,
            'recommendations': recommendations,
            'comparisons': comparisons
        }
        
        natural_analysis = self._generate_rendement_analysis(analysis_data)
        
        analysis = {
            'title': f"Analyse des rendements par {mode}",
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'patterns': patterns,
            'comparisons': comparisons,
            'risks': risks,
            'recommendations': recommendations,
            'natural_analysis': natural_analysis
        }
        
        return analysis
    
    def analyze_volume(self, data: pd.DataFrame, mode: str, selections: List[str]) -> Dict[str, Any]:
        """Analyse des volumes"""
        
        df_analysis = data.copy()
        
        # Mapping des modes vers les noms de colonnes réels
        mode_to_col = {
            "Couleur": "code_couleur",
            "Departement": "code_departement",  # Sans accent
            "Zone": "Zone",
            "Cepage": "code_cepage"  # Sans accent
        }
        
        col_name = mode_to_col.get(mode, mode)
        
        stats = {}
        patterns = {}
        recommendations = []
        
        for selection in selections:
            subset = df_analysis[df_analysis[col_name] == selection]
            
            if not subset.empty:
                volumes = subset['volume'].dropna()
                
                if len(volumes) > 0:
                    total_volume = volumes.sum()
                    mean_volume = volumes.mean()
                    
                    stats[selection] = {
                        'total': total_volume,
                        'mean': mean_volume,
                        'median': volumes.median(),
                        'std': volumes.std(),
                        'cv': (volumes.std() / mean_volume * 100) if mean_volume > 0 else 0,
                        'n_years': len(subset['annee'].unique())
                    }
                    
                    time_data = subset.groupby('annee')['volume'].sum().reset_index()
                    patterns[selection] = self._detect_patterns(time_data, 'volume')
        
        # Analyse croisée
        cross_analysis = {}
        if len(selections) > 1:
            totals = {k: v['total'] for k, v in stats.items()}
            total_global = sum(totals.values())
            for k, v in totals.items():
                cross_analysis[k] = {
                    'share': (v / total_global * 100) if total_global > 0 else 0,
                    'total': v
                }
        
        # Génération analyse simplifiée
        lines = []
        lines.append(f"**Analyse des volumes par {mode}**")
        lines.append("")
        
        for groupe, stat in stats.items():
            share = cross_analysis.get(groupe, {}).get('share', 0)
            lines.append(f"**{groupe}** :")
            lines.append(f"  • Volume total : **{stat['total']:,.0f} hl** ({share:.1f}% du total)")
            lines.append(f"  • Moyenne annuelle : **{stat['mean']:,.0f} hl**")
            if stat['cv'] > 30:
                lines.append(f"  • Forte variation (CV={stat['cv']:.0f}%)")
            lines.append("")
        
        natural_analysis = "\n".join(lines)
        
        analysis = {
            'title': f"Analyse des volumes par {mode}",
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'patterns': patterns,
            'cross_analysis': cross_analysis,
            'recommendations': recommendations,
            'natural_analysis': natural_analysis
        }
        
        return analysis
    
    def get_learning_summary(self) -> Dict:
        """Résumé de l'apprentissage"""
        n_analyses = len(self.memory['analyses'])
        n_feedback = len(self.memory['feedback'])
        
        return {
            'n_analyses': n_analyses,
            'n_feedback': n_feedback,
            'insights': [
                f"{n_analyses} analyses réalisées",
                f"{n_feedback} retours utilisateur"
            ] if n_analyses > 0 else ["Prêt à analyser"]
        }
    
    def _learn_from_feedback(self, analysis_id: str, feedback: Dict):
        """Apprend des retours"""
        self.memory['feedback'].append({
            'analysis_id': analysis_id,
            'feedback': feedback,
            'timestamp': datetime.now()
        })
        self._save_memory()