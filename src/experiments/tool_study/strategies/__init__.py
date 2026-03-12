"""Experimental query strategies for prompt retrieval."""

from .baseline import BaselineStrategy
from .source_balanced import SourceBalancedStrategy
from .decomposed_query import DecomposedQueryStrategy
from .dedup_aware import DedupAwareStrategy
from .cluster_diverse import ClusterDiverseStrategy

STRATEGIES = {
    "baseline": BaselineStrategy,
    "source_balanced": SourceBalancedStrategy,
    "decomposed": DecomposedQueryStrategy,
    "dedup_aware": DedupAwareStrategy,
    "cluster_diverse": ClusterDiverseStrategy,
}
