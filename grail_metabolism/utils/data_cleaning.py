import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import logging
from typing import List, Dict, Any, Tuple, Optional
import warnings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HighDiversityScaffoldSelector:
    """
    A scaffold-based molecular diversity selector optimized for maximum chemical diversity.

    This selector employs an inverse priority strategy that favors rare scaffolds over
    common ones to maximize structural diversity in molecular datasets. Instead of
    taking many molecules from popular scaffolds, it focuses on broad coverage of
    chemical space by prioritizing underrepresented structural motifs.

    Key Features:
    - Inverse priority: Rare scaffolds get higher selection limits
    - Adaptive limits: Selection limits scale with scaffold rarity
    - Diversity-first: Maximizes unique scaffold count rather than molecule count per scaffold
    """

    def __init__(self, target_size: int = 5000000, n_workers: int = None,
                 max_scaffold_molecules: int = 20,
                 rare_scaffold_boost: int = 5,
                 diversity_threshold: float = 0.7):
        """
        Initialize the diversity-optimized scaffold selector.

        Args:
            target_size: Total number of molecules to select
            n_workers: Number of parallel workers (uses CPU count - 1 if None)
            max_scaffold_molecules: Base limit for molecules per scaffold
            rare_scaffold_boost: Multiplier for rare scaffolds (increases diversity)
            diversity_threshold: Target scaffold-to-molecule ratio for quality control
        """
        self.target_size = target_size
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.max_scaffold_molecules = max_scaffold_molecules
        self.rare_scaffold_boost = rare_scaffold_boost
        self.diversity_threshold = diversity_threshold

    def optimized_selection(self, smiles_file: str, output_file: str) -> List[str]:
        """
        Execute the diversity-optimized molecular selection pipeline.

        The pipeline consists of three main stages:
        1. Scaffold distribution analysis to understand dataset composition
        2. Diversity-optimized selection planning with inverse priority strategy
        3. Execution of the selection plan with quality control

        Args:
            smiles_file: Path to input SMILES file
            output_file: Path for output selected molecules

        Returns:
            List of selected SMILES strings
        """
        logger.info("Starting diversity-optimized molecular selection pipeline")

        # Stage 1: Comprehensive scaffold distribution analysis
        scaffold_stats = self._analyze_scaffold_distribution(smiles_file)

        # Stage 2: Inverse priority selection strategy
        selection_plan = self._diversity_optimized_plan(scaffold_stats)

        # Stage 3: Plan execution with diversity preservation
        final_smiles = self._execute_diverse_selection(smiles_file, selection_plan)

        # Save results
        with open(output_file, 'w') as f:
            for smiles in final_smiles:
                f.write(smiles + '\n')

        logger.info(f"Selection complete: {len(final_smiles)} molecules saved to {output_file}")
        return final_smiles

    def _analyze_scaffold_distribution(self, smiles_file: str) -> Dict[str, Any]:
        """
        Analyze scaffold distribution to identify rarity patterns.

        This method processes the entire dataset to:
        - Count molecules per scaffold
        - Identify the rarity cutoff point (where 80% of molecules are contained)
        - Categorize scaffolds by size for strategic selection

        Args:
            smiles_file: Path to input SMILES file

        Returns:
            Dictionary containing scaffold statistics and distribution analysis
        """
        logger.info("Analyzing scaffold distribution patterns")

        scaffold_counts = Counter()
        total_molecules = 0

        with open(smiles_file, 'r') as f:
            for line in tqdm(f, desc="Processing scaffolds"):
                smiles = line.strip()
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and self._passes_filters(mol):
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffold_counts[scaffold_smiles] += 1
                        total_molecules += 1
                except:
                    continue

        # Identify rarity distribution
        sorted_scaffolds = scaffold_counts.most_common()
        cumulative = 0
        rare_cutoff = 0

        for i, (scaffold, count) in enumerate(sorted_scaffolds):
            cumulative += count
            if cumulative >= total_molecules * 0.8:
                rare_cutoff = i
                break

        return {
            'total_scaffolds': len(scaffold_counts),
            'total_molecules': total_molecules,
            'scaffold_counts': scaffold_counts,
            'rare_cutoff_index': rare_cutoff,
            'top_scaffolds': sorted_scaffolds[:1000],
            'rare_scaffolds': sorted_scaffolds[rare_cutoff:]
        }

    def _diversity_optimized_plan(self, stats: Dict[str, Any]) -> Dict[str, int]:
        """
        Create selection plan using inverse priority strategy.

        The strategy employs four-tier scaffold categorization:
        1. Super-rare (1-3 molecules): Take ALL molecules (100% preservation)
        2. Rare (4-20 molecules): Boosted selection (up to 5x base limit)
        3. Medium (21-100 molecules): Moderate selection (up to 2x base limit)
        4. Large (>100 molecules): Restricted selection (0.5x base limit)

        Args:
            stats: Scaffold distribution statistics from analysis phase

        Returns:
            Dictionary mapping scaffold SMILES to selection limits
        """
        logger.info("Creating diversity-optimized selection plan")

        scaffold_plan = {}
        scaffold_counts = stats['scaffold_counts']

        # Tier 1: Super-rare scaffolds (complete preservation)
        super_rare_count = 0
        for scaffold, count in scaffold_counts.items():
            if count <= 3:
                scaffold_plan[scaffold] = count  # Take all molecules
                super_rare_count += 1

        logger.info(f"Super-rare scaffolds (1-3 molecules): {super_rare_count}")

        # Tier 2: Rare scaffolds (boosted selection)
        rare_scaffolds = [(s, c) for s, c in scaffold_counts.items()
                          if 4 <= c <= 20 and s not in scaffold_plan]
        rare_scaffolds.sort(key=lambda x: x[1])  # Sort by rarity (ascending)

        for scaffold, count in rare_scaffolds[:50000]:
            scaffold_plan[scaffold] = min(count, self.max_scaffold_molecules * self.rare_scaffold_boost)

        # Tier 3: Medium scaffolds (moderate selection)
        medium_scaffolds = [(s, c) for s, c in scaffold_counts.items()
                            if 21 <= c <= 100 and s not in scaffold_plan]
        medium_scaffolds.sort(key=lambda x: x[1])

        for scaffold, count in medium_scaffolds[:20000]:
            scaffold_plan[scaffold] = min(count, self.max_scaffold_molecules * 2)

        # Tier 4: Large scaffolds (restricted selection)
        large_scaffolds = [(s, c) for s, c in scaffold_counts.items()
                           if c > 100 and s not in scaffold_plan]
        large_scaffolds.sort(key=lambda x: x[1])

        for scaffold, count in large_scaffolds[:5000]:
            scaffold_plan[scaffold] = min(count, self.max_scaffold_molecules // 2)

        total_planned = sum(scaffold_plan.values())
        logger.info(f"Selection plan covers {len(scaffold_plan)} scaffolds, {total_planned} molecules")

        return scaffold_plan

    def _execute_diverse_selection(self, smiles_file: str, selection_plan: Dict[str, int]) -> List[str]:
        """
        Execute the diversity-optimized selection plan.

        Processes the input file line by line, selecting molecules according to the
        scaffold-based limits defined in the selection plan.

        Args:
            smiles_file: Path to input SMILES file
            selection_plan: Dictionary of scaffold selection limits

        Returns:
            List of selected SMILES strings
        """
        logger.info("Executing diversity-optimized selection")

        selected_smiles = []
        scaffold_counts = {scaffold: 0 for scaffold in selection_plan.keys()}

        with open(smiles_file, 'r') as f:
            for line in tqdm(f, desc="Selecting molecules"):
                smiles = line.strip()
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and self._passes_filters(mol):
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold)

                        if (scaffold_smiles in scaffold_counts and
                                scaffold_counts[scaffold_smiles] < selection_plan[scaffold_smiles]):
                            selected_smiles.append(smiles)
                            scaffold_counts[scaffold_smiles] += 1
                except:
                    continue

        logger.info(f"Selected {len(selected_smiles)} molecules from {len(selection_plan)} scaffolds")

        # Fill remaining slots if needed
        if len(selected_smiles) < self.target_size:
            additional = self._select_additional_diverse(smiles_file, selected_smiles,
                                                         self.target_size - len(selected_smiles))
            selected_smiles.extend(additional)
            logger.info(f"Added {len(additional)} additional molecules to reach target size")

        return selected_smiles[:self.target_size]

    def _select_additional_diverse(self, smiles_file: str, existing_smiles: List[str],
                                   needed: int) -> List[str]:
        """
        Select additional diverse molecules to reach target size.

        Args:
            smiles_file: Path to input SMILES file
            existing_smiles: Already selected molecules
            needed: Number of additional molecules needed

        Returns:
            List of additional SMILES strings
        """
        existing_set = set(existing_smiles)
        additional = []

        with open(smiles_file, 'r') as f:
            for line in f:
                if len(additional) >= needed:
                    break
                smiles = line.strip()
                if smiles not in existing_set:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol and self._passes_filters(mol):
                            additional.append(smiles)
                    except:
                        continue

        return additional

    def _passes_filters(self, mol) -> bool:
        """
        Apply molecular filters with diversity-preserving thresholds.

        Uses relaxed filters to preserve chemical diversity while removing
        extreme outliers that might not be biologically relevant.

        Args:
            mol: RDKit molecule object

        Returns:
            Boolean indicating if molecule passes filters
        """
        try:
            heavy_atoms = mol.GetNumHeavyAtoms()
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)

            # Diversity-preserving filters (relaxed thresholds)
            return (3 <= heavy_atoms <= 100 and  # Extended range for diversity
                    30 <= mw <= 900 and  # Broader molecular weight
                    -6 <= logp <= 8)  # Extended LogP range
        except:
            return False


class DiversityAnalyzer:
    """
    Comprehensive molecular diversity analysis toolkit.

    Provides multiple methods for assessing chemical diversity including:
    - Scaffold-based diversity metrics
    - Molecular property distribution analysis
    - Chemical space visualization using t-SNE and PCA
    - Fingerprint-based similarity analysis
    - Comparative analysis against random baselines

    Diversity Calculation Methods:
    1. Scaffold Diversity: unique_scaffolds / total_molecules
    2. Fingerprint Diversity: 1 - average_pairwise_similarity
    3. Property Space Coverage: variance in molecular descriptors
    4. Cluster-based Metrics: silhouette scores and cluster distribution
    """

    def __init__(self, sample_size: int = 50000, random_state: int = 42):
        """
        Initialize the diversity analyzer.

        Args:
            sample_size: Maximum number of molecules to use for computationally intensive analyses
            random_state: Random seed for reproducible sampling
        """
        self.sample_size = sample_size
        self.random_state = random_state
        np.random.seed(random_state)

    def load_smiles(self, smiles_file: str) -> List[str]:
        """Load SMILES strings from file."""
        logger.info(f"Loading SMILES from {smiles_file}")
        with open(smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(smiles_list)} molecules")
        return smiles_list

    def comprehensive_analysis(self, smiles_list: List[str],
                               reference_smiles: Optional[List[str]] = None,
                               dataset_name: str = "analyzed_set") -> Dict[str, Any]:
        """
        Perform comprehensive diversity analysis on molecular dataset.

        Analysis includes:
        - Scaffold diversity and distribution
        - Molecular property space coverage
        - Chemical similarity metrics
        - Comparative analysis against random baseline
        - High-dimensional chemical space visualization

        Args:
            smiles_list: List of SMILES strings to analyze
            reference_smiles: Optional reference set for comparison
            dataset_name: Name for labeling results

        Returns:
            Dictionary containing comprehensive diversity metrics
        """
        logger.info("Starting comprehensive diversity analysis")

        # Create representative sample for computationally intensive analyses
        if len(smiles_list) > self.sample_size:
            sampled_smiles = np.random.choice(smiles_list, self.sample_size, replace=False).tolist()
        else:
            sampled_smiles = smiles_list

        results = {'dataset_name': dataset_name}

        # Core diversity analyses
        logger.info("Performing scaffold diversity analysis")
        scaffold_stats = self.analyze_scaffolds(sampled_smiles)
        results.update(scaffold_stats)

        logger.info("Analyzing molecular property distributions")
        property_stats = self.analyze_molecular_properties(sampled_smiles)
        results.update(property_stats)

        logger.info("Computing chemical similarity metrics")
        diversity_stats = self.analyze_chemical_diversity(sampled_smiles)
        results.update(diversity_stats)

        # Comparative analysis
        if reference_smiles:
            logger.info("Performing comparative analysis against reference")
            comparative_stats = self.compare_with_reference(sampled_smiles, reference_smiles)
            results.update(comparative_stats)

        # Advanced visualizations
        logger.info("Generating advanced visualizations")
        self.create_comprehensive_visualizations(sampled_smiles, results, reference_smiles)

        return results

    def analyze_scaffolds(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Analyze scaffold-based diversity metrics.

        Calculates:
        - Scaffold diversity ratio (unique_scaffolds / total_molecules)
        - Scaffold size distribution
        - Coverage metrics (scaffolds needed for 50%/80% coverage)
        - Top scaffold analysis

        Args:
            smiles_list: List of SMILES strings to analyze

        Returns:
            Dictionary containing scaffold diversity metrics
        """
        scaffolds = []
        valid_mols = 0

        for smiles in tqdm(smiles_list, desc="Analyzing scaffolds"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
                valid_mols += 1
            except:
                continue

        scaffold_counts = Counter(scaffolds)
        unique_scaffolds = len(scaffold_counts)

        # Size distribution analysis
        size_distribution = {
            'singleton': sum(1 for count in scaffold_counts.values() if count == 1),
            'small_2_5': sum(1 for count in scaffold_counts.values() if 2 <= count <= 5),
            'medium_6_20': sum(1 for count in scaffold_counts.values() if 6 <= count <= 20),
            'large_21_100': sum(1 for count in scaffold_counts.values() if 21 <= count <= 100),
            'very_large_100+': sum(1 for count in scaffold_counts.values() if count > 100)
        }

        top_scaffolds = scaffold_counts.most_common(10)

        return {
            'total_molecules_analyzed': valid_mols,
            'unique_scaffolds': unique_scaffolds,
            'scaffold_diversity_ratio': unique_scaffolds / valid_mols if valid_mols > 0 else 0,
            'avg_molecules_per_scaffold': valid_mols / unique_scaffolds if unique_scaffolds > 0 else 0,
            'scaffold_size_distribution': size_distribution,
            'top_scaffolds': top_scaffolds,
            'scaffold_coverage': self._calculate_scaffold_coverage(scaffold_counts)
        }

    def _calculate_scaffold_coverage(self, scaffold_counts: Counter) -> Dict[str, float]:
        """Calculate scaffold coverage metrics."""
        total_molecules = sum(scaffold_counts.values())
        sorted_scaffolds = scaffold_counts.most_common()

        # Find scaffolds needed for 50% and 80% coverage
        cumulative_count = 0
        scaffolds_50, scaffolds_80 = 0, 0

        for i, (scaffold, count) in enumerate(sorted_scaffolds):
            cumulative_count += count
            if scaffolds_50 == 0 and cumulative_count >= total_molecules * 0.5:
                scaffolds_50 = i + 1
            if cumulative_count >= total_molecules * 0.8:
                scaffolds_80 = i + 1
                break

        return {
            'scaffolds_for_50_percent': scaffolds_50,
            'scaffolds_for_80_percent': scaffolds_80,
            'percent_covered_by_top_100': sum(count for _, count in sorted_scaffolds[:100]) / total_molecules * 100
        }

    def analyze_molecular_properties(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Analyze distribution of key molecular properties.

        Properties analyzed:
        - Molecular weight, LogP, heavy atom count
        - Hydrogen bond donors/acceptors
        - Rotatable bonds, TPSA, aromatic rings
        - Fraction of sp3 carbons

        Args:
            smiles_list: List of SMILES strings to analyze

        Returns:
            Dictionary containing property distribution statistics
        """
        properties = {
            'molecular_weight': [], 'logp': [], 'num_h_donors': [], 'num_h_acceptors': [],
            'num_rotatable_bonds': [], 'tpsa': [], 'num_heavy_atoms': [], 'fraction_csp3': []
        }

        valid_mols = 0
        for smiles in tqdm(smiles_list, desc="Analyzing properties"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                properties['molecular_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Descriptors.MolLogP(mol))
                properties['num_h_donors'].append(Descriptors.NumHDonors(mol))
                properties['num_h_acceptors'].append(Descriptors.NumHAcceptors(mol))
                properties['num_rotatable_bonds'].append(Descriptors.NumRotatableBonds(mol))
                properties['tpsa'].append(Descriptors.TPSA(mol))
                properties['num_heavy_atoms'].append(mol.GetNumHeavyAtoms())
                properties['fraction_csp3'].append(Descriptors.FractionCsp3(mol))

                valid_mols += 1
            except:
                continue

        # Calculate comprehensive statistics
        stats = {}
        for prop_name, values in properties.items():
            if values:
                stats[f'{prop_name}_mean'] = np.mean(values)
                stats[f'{prop_name}_std'] = np.std(values)
                stats[f'{prop_name}_min'] = np.min(values)
                stats[f'{prop_name}_max'] = np.max(values)
                stats[f'{prop_name}_q25'] = np.percentile(values, 25)
                stats[f'{prop_name}_q75'] = np.percentile(values, 75)

        stats['property_analysis_molecules'] = valid_mols
        return stats

    def analyze_chemical_diversity(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Analyze chemical diversity using fingerprint-based metrics.

        Methods:
        - Pairwise Tanimoto similarity distribution
        - Clustering-based diversity assessment
        - Fingerprint density analysis

        Args:
            smiles_list: List of SMILES strings to analyze

        Returns:
            Dictionary containing chemical diversity metrics
        """
        logger.info("Generating molecular fingerprints for diversity analysis")

        sample_size = min(5000, len(smiles_list))
        sample_smiles = np.random.choice(smiles_list, sample_size, replace=False).tolist()

        fingerprints = []
        valid_smiles = []

        for smiles in tqdm(sample_smiles, desc="Generating fingerprints"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(fp)
                valid_smiles.append(smiles)
            except:
                continue

        if len(fingerprints) < 100:
            logger.warning("Insufficient valid fingerprints for diversity analysis")
            return {'diversity_analysis': 'insufficient_data'}

        # Similarity analysis
        similarity_stats = self._calculate_similarity_statistics(fingerprints)

        # Clustering analysis
        fingerprint_arrays = []
        for fp in fingerprints:
            arr = np.zeros((1024,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprint_arrays.append(arr)

        fingerprint_arrays = np.array(fingerprint_arrays)
        clustering_stats = self._analyze_clustering_diversity(fingerprint_arrays)

        return {
            'fingerprints_analyzed': len(fingerprints),
            **similarity_stats,
            **clustering_stats
        }

    def _calculate_similarity_statistics(self, fingerprints: List) -> Dict[str, Any]:
        """Calculate fingerprint similarity statistics."""
        logger.info("Computing pairwise similarity distribution")

        n_pairs = min(10000, len(fingerprints) * 10)
        similarities = []

        for _ in tqdm(range(n_pairs), desc="Calculating similarities"):
            i, j = np.random.choice(len(fingerprints), 2, replace=False)
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(similarity)

        similarities = np.array(similarities)

        return {
            'mean_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'similarity_std': np.std(similarities),
            'similarity_q95': np.percentile(similarities, 95),
            'diversity_score': 1 - np.mean(similarities)  # Primary diversity metric
        }

    def _analyze_clustering_diversity(self, fingerprints: np.ndarray) -> Dict[str, Any]:
        """Analyze diversity through clustering approach."""
        n_clusters = min(50, len(fingerprints) // 10)

        if n_clusters < 2:
            return {'clustering_analysis': 'insufficient_data'}

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(fingerprints)

        # Silhouette score for clustering quality
        try:
            sample_size = min(1000, len(fingerprints))
            sample_indices = np.random.choice(len(fingerprints), sample_size, replace=False)
            silhouette_avg = silhouette_score(fingerprints[sample_indices], labels[sample_indices])
        except:
            silhouette_avg = -1

        cluster_sizes = Counter(labels)

        return {
            'n_clusters_used': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_size_std': np.std(list(cluster_sizes.values())),
            'clusters_with_1_molecule': sum(1 for count in cluster_sizes.values() if count == 1),
            'avg_cluster_size': len(fingerprints) / n_clusters
        }

    def compare_with_reference(self, test_smiles: List[str], reference_smiles: List[str]) -> Dict[str, Any]:
        """
        Compare test set diversity against reference set.

        Creates a random sample from reference set of same size as test set
        and compares key diversity metrics.

        Args:
            test_smiles: Test set SMILES strings
            reference_smiles: Reference set SMILES strings

        Returns:
            Dictionary containing comparative diversity metrics
        """
        logger.info("Comparing against reference set")

        # Create comparable random sample from reference
        sample_size = min(len(test_smiles), len(reference_smiles))
        reference_sample = np.random.choice(reference_smiles, sample_size, replace=False).tolist()

        # Analyze both sets
        test_analysis = self.analyze_scaffolds(test_smiles)
        reference_analysis = self.analyze_scaffolds(reference_sample)

        # Calculate comparison metrics
        scaffold_ratio_test = test_analysis['scaffold_diversity_ratio']
        scaffold_ratio_ref = reference_analysis['scaffold_diversity_ratio']

        return {
            'reference_scaffold_diversity': scaffold_ratio_ref,
            'test_scaffold_diversity': scaffold_ratio_test,
            'diversity_improvement': (scaffold_ratio_test - scaffold_ratio_ref) / scaffold_ratio_ref * 100,
            'test_unique_scaffolds': test_analysis['unique_scaffolds'],
            'reference_unique_scaffolds': reference_analysis['unique_scaffolds']
        }

    def create_comprehensive_visualizations(self, smiles_list: List[str],
                                            results: Dict[str, Any],
                                            reference_smiles: Optional[List[str]] = None):
        """
        Create comprehensive visualizations for diversity analysis.

        Generates:
        - Scaffold distribution plots
        - Molecular property distributions
        - Chemical space visualizations (t-SNE, PCA)
        - Diversity comparison plots
        - Fingerprint similarity distributions

        Args:
            smiles_list: SMILES strings for visualization
            results: Diversity analysis results
            reference_smiles: Optional reference set for comparison
        """
        logger.info("Creating comprehensive visualizations")

        self._plot_scaffold_distribution(results)
        self._plot_molecular_properties(smiles_list, results)
        self._plot_chemical_space(smiles_list, reference_smiles)
        self._plot_diversity_comparison(results)
        self._generate_comprehensive_report(results)

    def _plot_scaffold_distribution(self, results: Dict[str, Any]):
        """Create scaffold distribution visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Scaffold size distribution (bar plot)
        size_dist = results.get('scaffold_size_distribution', {})
        categories = ['Singleton', '2-5', '6-20', '21-100', '100+']
        counts = [size_dist.get('singleton', 0),
                  size_dist.get('small_2_5', 0),
                  size_dist.get('medium_6_20', 0),
                  size_dist.get('large_21_100', 0),
                  size_dist.get('very_large_100+', 0)]

        axes[0, 0].bar(categories, counts, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Scaffold Size Distribution')
        axes[0, 0].set_ylabel('Number of Scaffolds')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Top scaffolds
        top_scaffolds = results.get('top_scaffolds', [])[:10]
        if top_scaffolds:
            scaffolds, counts = zip(*top_scaffolds)
            axes[0, 1].bar(range(len(scaffolds)), counts, color='lightcoral')
            axes[0, 1].set_title('Top 10 Most Frequent Scaffolds')
            axes[0, 1].set_ylabel('Molecule Count')
            axes[0, 1].set_xticks(range(len(scaffolds)))
            axes[0, 1].set_xticklabels([f'Scaffold {i + 1}' for i in range(len(scaffolds))])

        # Coverage analysis
        coverage = results.get('scaffold_coverage', {})
        coverage_data = [coverage.get('scaffolds_for_50_percent', 0),
                         coverage.get('scaffolds_for_80_percent', 0)]
        axes[1, 0].bar(['50% Coverage', '80% Coverage'], coverage_data, color='lightgreen')
        axes[1, 0].set_title('Scaffolds Required for Coverage')
        axes[1, 0].set_ylabel('Number of Scaffolds')

        # Diversity metrics summary
        metrics_text = f"""
        Total Molecules: {results.get('total_molecules_analyzed', 0):,}
        Unique Scaffolds: {results.get('unique_scaffolds', 0):,}
        Diversity Ratio: {results.get('scaffold_diversity_ratio', 0):.3f}
        Avg per Scaffold: {results.get('avg_molecules_per_scaffold', 0):.1f}
        """
        axes[1, 1].text(0.1, 0.7, metrics_text, fontsize=12, va='center', linespacing=1.5)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Diversity Metrics Summary')

        plt.tight_layout()
        plt.savefig('scaffold_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_molecular_properties(self, smiles_list: List[str], results: Dict[str, Any]):
        """Create molecular property distribution plots."""
        sample_size = min(2000, len(smiles_list))
        sample_smiles = np.random.choice(smiles_list, sample_size, replace=False).tolist()

        properties = {
            'Molecular Weight': [],
            'LogP': [],
            'Heavy Atoms': [],
            'TPSA': [],
            'Rotatable Bonds': []
        }

        for smiles in sample_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    properties['Molecular Weight'].append(Descriptors.MolWt(mol))
                    properties['LogP'].append(Descriptors.MolLogP(mol))
                    properties['Heavy Atoms'].append(mol.GetNumHeavyAtoms())
                    properties['TPSA'].append(Descriptors.TPSA(mol))
                    properties['Rotatable Bonds'].append(Descriptors.NumRotatableBonds(mol))
            except:
                continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for i, (prop_name, values) in enumerate(properties.items()):
            if i >= len(axes):
                break
            if values:
                axes[i].hist(values, bins=50, alpha=0.7, color='steelblue',
                             edgecolor='black', density=True)
                axes[i].set_title(f'Distribution of {prop_name}')
                axes[i].set_xlabel(prop_name)
                axes[i].set_ylabel('Density')

                # Add statistical annotations
                mean_val = np.mean(values)
                std_val = np.std(values)
                axes[i].axvline(mean_val, color='red', linestyle='--',
                                label=f'Mean: {mean_val:.1f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle=':')
                axes[i].axvline(mean_val - std_val, color='orange', linestyle=':')
                axes[i].legend()

        # Remove empty subplots
        for i in range(len(properties), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig('molecular_property_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_chemical_space(self, smiles_list: List[str],
                             reference_smiles: Optional[List[str]] = None):
        """
        Create chemical space visualizations using t-SNE and PCA.

        Projects molecular fingerprints into 2D space for visualization
        of chemical space coverage and clustering patterns.
        """
        logger.info("Generating chemical space visualizations")

        # Prepare fingerprints
        sample_size = min(2000, len(smiles_list))
        sample_smiles = np.random.choice(smiles_list, sample_size, replace=False).tolist()

        fingerprints = []
        valid_smiles = []

        for smiles in tqdm(sample_smiles, desc="Preparing fingerprints for visualization"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                    arr = np.zeros(512, dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fingerprints.append(arr)
                    valid_smiles.append(smiles)
            except:
                continue

        if len(fingerprints) < 100:
            logger.warning("Insufficient fingerprints for chemical space visualization")
            return

        fingerprints = np.array(fingerprints)

        # Prepare reference fingerprints if provided
        reference_fingerprints = None
        if reference_smiles:
            ref_sample_size = min(1000, len(reference_smiles))
            ref_sample = np.random.choice(reference_smiles, ref_sample_size, replace=False).tolist()
            reference_fingerprints = []

            for smiles in ref_sample:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                        arr = np.zeros(512, dtype=np.float32)
                        DataStructs.ConvertToNumpyArray(fp, arr)
                        reference_fingerprints.append(arr)
                except:
                    continue

            if len(reference_fingerprints) < 50:
                reference_fingerprints = None

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(fingerprints)

        scatter = axes[0].scatter(tsne_result[:, 0], tsne_result[:, 1],
                                  alpha=0.6, s=20, c='blue', label='Selected Set')
        axes[0].set_title('t-SNE Projection of Chemical Space')
        axes[0].set_xlabel('t-SNE Component 1')
        axes[0].set_ylabel('t-SNE Component 2')

        # Add reference set if available
        if reference_fingerprints is not None:
            ref_fingerprints_array = np.array(reference_fingerprints)
            # Combine for consistent t-SNE transformation
            combined_fps = np.vstack([fingerprints, ref_fingerprints_array])
            combined_tsne = tsne.fit_transform(combined_fps)

            # Split back
            tsne_selected = combined_tsne[:len(fingerprints)]
            tsne_reference = combined_tsne[len(fingerprints):]

            axes[0].clear()
            axes[0].scatter(tsne_selected[:, 0], tsne_selected[:, 1],
                            alpha=0.6, s=20, c='blue', label='Selected Set')
            axes[0].scatter(tsne_reference[:, 0], tsne_reference[:, 1],
                            alpha=0.6, s=20, c='red', label='Reference Set')
            axes[0].set_title('t-SNE: Selected vs Reference Sets')
            axes[0].legend()

        # PCA visualization
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(fingerprints)

        axes[1].scatter(pca_result[:, 0], pca_result[:, 1],
                        alpha=0.6, s=20, c='green')
        axes[1].set_title('PCA Projection of Chemical Space')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

        plt.tight_layout()
        plt.savefig('chemical_space_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_diversity_comparison(self, results: Dict[str, Any]):
        """Create diversity comparison visualizations."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scaffold diversity comparison
        if 'reference_scaffold_diversity' in results:
            diversity_values = [
                results['reference_scaffold_diversity'],
                results['test_scaffold_diversity']
            ]
            labels = ['Reference Set', 'Selected Set']

            bars = axes[0].bar(labels, diversity_values, color=['lightgray', 'lightblue'])
            axes[0].set_ylabel('Scaffold Diversity Ratio')
            axes[0].set_title('Scaffold Diversity Comparison')

            # Add value labels on bars
            for bar, value in zip(bars, diversity_values):
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom')

        # Unique scaffolds comparison
        if 'reference_unique_scaffolds' in results:
            scaffold_counts = [
                results['reference_unique_scaffolds'],
                results['test_unique_scaffolds']
            ]
            labels = ['Reference Set', 'Selected Set']

            bars = axes[1].bar(labels, scaffold_counts, color=['lightgray', 'lightcoral'])
            axes[1].set_ylabel('Unique Scaffolds')
            axes[1].set_title('Structural Diversity Comparison')

            # Add value labels on bars
            for bar, value in zip(bars, scaffold_counts):
                axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{value:,}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('diversity_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive diversity analysis report."""
        report = []
        report.append("=" * 70)
        report.append("COMPREHENSIVE MOLECULAR DIVERSITY ANALYSIS REPORT")
        report.append("=" * 70)

        # Dataset information
        report.append(f"\nDATASET: {results.get('dataset_name', 'Unknown')}")
        report.append("-" * 40)

        # Scaffold diversity section
        report.append("\n1. SCAFFOLD DIVERSITY ANALYSIS:")
        report.append(f"   Total molecules analyzed: {results.get('total_molecules_analyzed', 0):,}")
        report.append(f"   Unique scaffolds identified: {results.get('unique_scaffolds', 0):,}")
        report.append(f"   Scaffold diversity ratio: {results.get('scaffold_diversity_ratio', 0):.3f}")
        report.append(f"   Average molecules per scaffold: {results.get('avg_molecules_per_scaffold', 0):.1f}")

        coverage = results.get('scaffold_coverage', {})
        report.append(f"   Scaffolds for 50% coverage: {coverage.get('scaffolds_for_50_percent', 0):,}")
        report.append(f"   Scaffolds for 80% coverage: {coverage.get('scaffolds_for_80_percent', 0):,}")
        report.append(f"   Top 100 scaffolds coverage: {coverage.get('percent_covered_by_top_100', 0):.1f}%")

        # Chemical diversity section
        report.append("\n2. CHEMICAL DIVERSITY METRICS:")
        report.append(f"   Mean pairwise similarity: {results.get('mean_similarity', 0):.3f}")
        report.append(f"   Diversity score (1 - similarity): {results.get('diversity_score', 0):.3f}")

        if 'silhouette_score' in results:
            report.append(f"   Clustering silhouette score: {results.get('silhouette_score', 0):.3f}")

        # Molecular properties section
        report.append("\n3. MOLECULAR PROPERTY DISTRIBUTION:")
        props = ['molecular_weight', 'logp', 'num_heavy_atoms', 'tpsa']
        prop_names = ['Molecular Weight', 'LogP', 'Heavy Atoms', 'TPSA']

        for prop, name in zip(props, prop_names):
            mean_val = results.get(f'{prop}_mean', 0)
            std_val = results.get(f'{prop}_std', 0)
            report.append(f"   {name}: {mean_val:.1f} ± {std_val:.1f}")

        # Comparative analysis section
        if 'reference_scaffold_diversity' in results:
            report.append("\n4. COMPARATIVE ANALYSIS:")
            report.append(f"   Reference diversity: {results.get('reference_scaffold_diversity', 0):.3f}")
            report.append(f"   Selected set diversity: {results.get('test_scaffold_diversity', 0):.3f}")
            report.append(f"   Diversity improvement: {results.get('diversity_improvement', 0):.1f}%")

        # Quality assessment
        report.append("\n5. DIVERSITY QUALITY ASSESSMENT:")
        diversity_score = results.get('diversity_score', 0)
        scaffold_ratio = results.get('scaffold_diversity_ratio', 0)

        if diversity_score > 0.9 and scaffold_ratio > 0.8:
            assessment = "EXCELLENT - High chemical diversity with broad scaffold coverage"
        elif diversity_score > 0.8 and scaffold_ratio > 0.6:
            assessment = "GOOD - Well-balanced diversity with good scaffold representation"
        elif diversity_score > 0.7 and scaffold_ratio > 0.4:
            assessment = "MODERATE - Acceptable diversity with room for improvement"
        elif diversity_score > 0.6 and scaffold_ratio > 0.3:
            assessment = "LOW - Limited diversity, consider re-selection with different parameters"
        else:
            assessment = "POOR - Insufficient diversity for robust machine learning applications"

        report.append(f"   Overall assessment: {assessment}")
        report.append("=" * 70)

        # Save report
        with open('comprehensive_diversity_report.txt', 'w') as f:
            f.write('\n'.join(report))

        # Print to console
        print('\n'.join(report))


# Example usage and demonstration
def demonstrate_diversity_selection():
    """Demonstrate the complete diversity selection and analysis pipeline."""

    # Initialize selector with diversity-optimized parameters
    selector = HighDiversityScaffoldSelector(
        target_size=5000000,
        max_scaffold_molecules=20,
        rare_scaffold_boost=5,
        diversity_threshold=0.7
    )

    # Execute diversity-optimized selection
    selected_smiles = selector.optimized_selection(
        smiles_file="pubchem_10M.smi",
        output_file="high_diversity_5M.smi"
    )

    # Load reference set for comparison
    analyzer = DiversityAnalyzer(sample_size=50000, random_state=42)
    reference_smiles = analyzer.load_smiles("pubchem_10M.smi")

    # Perform comprehensive analysis
    analysis_results = analyzer.comprehensive_analysis(
        smiles_list=selected_smiles,
        reference_smiles=reference_smiles,
        dataset_name="HighDiversity_5M"
    )

    return analysis_results
