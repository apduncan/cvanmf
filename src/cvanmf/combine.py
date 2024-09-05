from __future__ import annotations

import itertools
import logging
import math
from collections import Counter, namedtuple
from typing import Union, Tuple, Iterable, List, Set, Literal, Optional, Dict, \
    Callable, Any

import networkx as nx
import numpy as np
import pandas as pd
import plotnine
from scipy.stats import kruskal
from sklearn.metrics.pairwise import cosine_similarity

from cvanmf.denovo import Decomposition
from cvanmf.reapply import match_identical, _reapply_model
from cvanmf.stability import Comparable, align_signatures, match_signatures, \
    get_signatures_from_comparable


# Alias for types which hold signatures that can be compared


def combine_signatures(
        signatures: Iterable[Comparable],
        x: pd.DataFrame = None,
        merge_threshold: float = 0.9,
        split_threshold: float = 0.9,
        prune_low_support: bool = True,
        low_support_threshold: Union[int, float] = 0.2,
        low_support_alpha: float = 0.05
) -> Decomposition:
    """Combine signatures into a non-redundant set."""

    # 1. Cosine similarity between all
    logging.info("Concatenating signatures from models")
    aligned_sigs: List[pd.DataFrame] = align_signatures(signatures)
    cat_signatures: pd.DataFrame = pd.concat(
        aligned_sigs,
        axis=1
    )
    merged_sigs: pd.DataFrame = cat_signatures
    # Keep identifying and merging cliques until no cliques |c| > 1 remain
    while True:
        cos: pd.DataFrame = cosine_similarity(merged_sigs.T)

        # 2. Maximal clique detection
        # This might benefit from being replaced with some near-clique methods
        # Could replace with some clustering method instead if this seems too
        # crude
        # Apply threshold to get adjacency matrix, convert to graph, B-K to find
        # cliques
        logging.info("Identifying cliques in similarity graph")
        adj: pd.DataFrame = cos > merge_threshold
        g = nx.from_numpy_array(adj)
        cliques = list(nx.find_cliques(g))
        logging.info("Found %s cliques, merging to single signatures",
                     len(cliques))

        # If all cliques are a single node, terminate merging
        if max(len(x) for x in cliques) < 2:
            break

        # 3. Merging
        # Make a new dataframe with the merged signatures
        # After making, remove all deleted signatures
        # Concatenate on merged signatures
        merged_sigs_i: pd.DataFrame = pd.concat(
            [merged_sigs.iloc[:, c].mean(axis=1) for c in cliques],
            axis=1
        )
        merged_sigs = pd.concat(
            [merged_sigs_i,
             merged_sigs.iloc[:,
             [x for x in range(merged_sigs.shape[1])
              if x not in list(itertools.chain.from_iterable(cliques))]
             ]],
            axis=1
        )

    # 4a. Fit each signature S_a to S_{-a} to and see if good similarity
    # (meaning it can be described well as a mix as some other signatures)
    # and can probably be removed
    logging.info("Pruning signatures which are adequately described as a mix of"
                 "other signatures")
    sig_remove: List[int] = []
    for i in range(merged_sigs.shape[1]):
        retain_sigs: List[int] = [x for x in range(merged_sigs.shape[1])
                                  if x not in sig_remove and x != i]
        reduced_sigs: pd.DataFrame = merged_sigs.iloc[:, retain_sigs]
        reduced_model: Decomposition = _reapply_model(
            y=merged_sigs.iloc[:, i].to_frame(),
            w=reduced_sigs,
            input_validation=lambda x: x,
            feature_match=match_identical,
            colors=None
        )
        if reduced_model.model_fit.iloc[0] > split_threshold:
            sig_remove.append(i)
    logging.info("Removed %s signatures which are linear combinations"
                 "of others", len(sig_remove))
    merged_sigs = merged_sigs.iloc[:,
                  [x for x in range(merged_sigs.shape[1])
                   if x not in sig_remove]]

    if not prune_low_support:
        return merged_sigs

    # 5. Remove signatures with low support which do not significantly
    # change model fit when removed
    # Determine number of models which contained something similar to
    # this signature, using the merge threshold
    support: pd.Series = pd.concat(
        [pd.Series(
            (cosine_similarity(merged_sigs.T, x.T) > merge_threshold
             ).any(axis=1)
        )
            for x in aligned_sigs],
        axis=1
    ).sum(axis=1)
    # support is now a series with the number of models which contain a
    # signature similar to this
    req_support: int  # If signature in this or more models, is well supported
    match low_support_threshold:
        case int():
            req_support = low_support_threshold
        case float():
            req_support = math.ceil(len(aligned_sigs) * low_support_threshold)
        case _:
            raise TypeError('low_support_threshold should be int or float')
    req_support = max(1, min(req_support, len(aligned_sigs)))
    low_support_sigs: pd.DataFrame = support[support < req_support]
    # Test the effect of removing the low support signature from a model
    # containing only the well supported signatures and this one
    good_sigs: pd.DataFrame = merged_sigs.iloc[:,
                              [x for x in support.index
                               if x not in low_support_sigs.index]]
    reject_low: List[int] = []
    for poor_sig, support in low_support_sigs.items():
        i_sigs: pd.DataFrame = pd.concat(
            [merged_sigs.iloc[:, poor_sig],
             good_sigs],
            axis=1,
        )
        # Rename to avoid undesired behaviour with dropping column by index
        i_sigs.columns = [f's{i}' for i in i_sigs.columns]
        loo_modelfit: List[pd.Series] = [
            _reapply_model(y=x,
                           w=i_sigs.drop(columns=[i]),
                           colors=None,
                           input_validation=lambda x: x,
                           feature_match=match_identical).model_fit
            for i in i_sigs.columns
        ]
        h, p = kruskal(loo_modelfit[0],
                       pd.concat(loo_modelfit[1:]))
        if (loo_modelfit[0].mean() > pd.concat(loo_modelfit[1:]).mean() and
                p <= low_support_alpha):
            # Reject this signature
            reject_low.append(poor_sig)
    merged_sigs = merged_sigs.iloc[:, [x for x in range(merged_sigs.shape[1])
                                       if x not in reject_low]]

    return merged_sigs


class Signature(pd.Series):
    """One signature which is being combined."""

    def __init__(self, model: Optional[Model] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    @property
    def model(self) -> Optional[Model]:
        return self.__model

    @model.setter
    def model(self, model: Optional[Model] = None) -> None:
        self.__model: Model = model

    @staticmethod
    def from_comparable(c: Comparable) -> List[Signature]:
        return [Signature(data=col) for _, col
                in get_signatures_from_comparable(c).T.iterrows()]

    @staticmethod
    def mds(signatures: Union[Iterable[Signature], pd.DataFrame],
            **kwargs) -> pd.DataFrame:
        """Perform MDS ordination of a set of signatures.

        Positions signatures in an n-dimensional space using cosine distance.
        Using the sklearn MDS implementation, so all arguments to the MDS
        constructor can be passed in kwargs. Generally useful ones are
        n_components for number of dimensions, metric for metric or non-metric
        MDS.

        :param signatures: Iterable of signatures, or a DataFrame of signatures
        :param n: Dimensions to use in NMDS
        :returns: DataFrame with signatures on rows and dimensions coordinates
        on columns.
        """
        from sklearn.manifold import MDS

        # Make a DataFrame is input is not already one
        sig_df: pd.DataFrame
        if not isinstance(signatures, pd.DataFrame):
            # Convert signatures to a list so we can guarantee ordered access
            # There's an outside chance we might be passed a list
            signatures = list(signatures)
            sig_df = pd.concat(signatures, axis=1)
        else:
            sig_df = signatures

        # Make cosine distance as input
        cos_dist: np.ndarray = 1.0 - cosine_similarity(sig_df.T)

        mds: MDS = MDS(dissimilarity="precomputed", **kwargs)
        transformed: np.ndarray = mds.fit_transform(cos_dist)
        transformed_df: pd.DataFrame = pd.DataFrame(
            transformed,
            index=sig_df.columns,
            columns=[f'dim_{i}' for i in range(1, mds.n_components + 1)]
        )

        if not isinstance(signatures, pd.DataFrame):
            # Attempt to attached cohort and model id as metadata where set
            transformed_df['cohort'] = [
                'None' if x.model.cohort is None else x.model.cohort.name
                for x in signatures
            ]
            transformed_df['model'] = [id(x.model) for x in signatures]
            transformed_df['model_rank'] = [len(x.model.signatures)
                                            for x in signatures]
        return transformed_df

    def __repr__(self) -> str:
        return f'Signature[id={id(self)}]'

    def __hash__(self) -> int:
        """Uses object memory address as hash value."""
        return id(self)


class Model:
    """One decomposition which is being combined."""

    def __init__(self,
                 signatures: Optional[Iterable[Signature]] = None,
                 cohort: Optional[Cohort] = None):
        self.signatures = signatures
        self.__cohort: Cohort = cohort

    @property
    def signatures(self) -> List[Signature]:
        return self.__signatures

    @signatures.setter
    def signatures(self, signatures: Iterable[Signature]) -> None:
        self.__signatures = list() if signatures is None else list(signatures)

    @property
    def cohort(self) -> Cohort:
        return self.__cohort

    @cohort.setter
    def cohort(self, cohort: Cohort) -> None:
        self.__cohort = cohort

    def add_signatures(self, signatures: Iterable[Signature]) -> Model:
        """Add signatures and set them to refer to this model."""

        for x in signatures:
            x.model = self
        self.signatures += list(signatures)
        return self

    @staticmethod
    def from_comparable(comparable: Comparable) -> Model:
        sigs: List[Signature] = Signature.from_comparable(comparable)
        model: Model = Model()
        model.add_signatures(sigs)
        return model


    def __repr__(self) -> str:
        return (f'Model[nsigs={len(self.signatures)}, '
                f'cohort='
                f'{"None" if self.cohort is None else self.cohort.name}]')


class Cohort:

    def __init__(self,
                 name: str,
                 models: Optional[List[Model]] = None,
                 x: Optional[pd.DataFrame] = None) -> None:
        self.name = name
        self.models = models
        self.x = x

    @property
    def name(self) -> str:
        return self.__name if self.__name is not None else str(id(self))

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @property
    def models(self) -> List[Model]:
        return self.__models

    @models.setter
    def models(self, models: Optional[List[Model]]) -> None:
        self.__models = [] if models is None else models

    @property
    def x(self) -> Optional[pd.DataFrame]:
        return self.__x

    @x.setter
    def x(self, x: Optional[pd.DataFrame]) -> None:
        self.__x: pd.DataFrame = x

    @property
    def signatures(self) -> List[Signature]:
        return list(itertools.chain.from_iterable(
            x.signatures for x in self.models))

    @staticmethod
    def from_comparables(
            comparables: Iterable[Comparable],
            name: str = "",
            x: Optional[pd.DataFrame] = None
    ) -> Cohort:
        models: List[Model] = [Model.from_comparable(x) for x in comparables]
        cohort: Cohort = Cohort(name=name, models=models, x=x)
        for m in cohort.models:
            m.cohort = cohort
        return cohort

    def add_models(self, models: Iterable[Model]) -> Cohort:
        for m in models:
            m.cohort = self
        self.models += list(models)
        return self

    def __repr__(self):
        return (f'Cohort[nmodels={len(self.models)}, '
                f'nsigs={len(self.signatures)}]')


class Cluster:

    def __init__(self,
                 signatures: Optional[Iterable[Signature]] = None,
                 label: Optional[str] = None):
        self.__signatures: List[Signature] = (
            list(signatures) if signatures is not None else [])
        self.label = label

    @property
    def signatures(self) -> List[Signature]:
        return self.__signatures

    @property
    def mean_signature(self) -> pd.Series:
        """Return the mean signature for this cluster."""

        return pd.concat(self.signatures, axis=1).mean(axis=1)

    @property
    def member_data(self) -> pd.DataFrame:
        """Data from all cohorts which any member signature originates from."""

        distinct_cohorts: Set[Cohort] = set(
            x.model.cohort for x in self.signatures
        )
        merged_df: pd.DataFrame = pd.concat([x.x for x in distinct_cohorts],
                                            axis=1)
        # If there are duplicate columns removed them but emit warning
        orig_shape: Tuple[int, int] = merged_df.shape
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        if orig_shape != merged_df.shape:
            logging.warning("Sample names duplicated between cohorts. If these "
                            "are distinct samples give them different names, "
                            "otherwise they will be treated as one sample "
                            "during signature combining.")
        return merged_df

    @property
    def label(self) -> str:
        """Label for this cluster in plots."""
        return f'Cluster_{id(self)}' if self.__label is None else self.__label

    @label.setter
    def label(self, label: str) -> None:
        """Setter for label."""
        self.__label: str = label

    def support(self,
                type: Literal["signature", "model", "cohort"] = "signature",
                samples: bool = False
                ) -> int:
        """The support for this cluster.

        We might want to place more confidence in signatures which appear more
        frequently, either having more similar signatures overall, appearing
        in more models, or appearing in more cohorts. This function provides
        either the number of signatures in the cluster, the number of models
        the cluster members originate from (likely to be the same as number of
        signatures), or the number of cohorts the signature is found in.
        Alternatively, we can express this as the total number of samples for
        the model or cohort case, for when some cohorts are small and we might
        want to weight those signatures from large cohorts.

        :param type: What to count. One of 'signature', 'model', 'cohort'
        :param samples: Count the number of samples for model or cohort
            rather than treating each as one.
        """

        if type == "signature" or type == "signatures":
            return len(self.signatures)
        elif type == "model" or type == "models":
            distinct_models = set(x.model for x in self.signatures)
            return sum(m.cohort.x.shape[1] for m in distinct_models) \
                if samples else len(distinct_models)
        elif type == "cohort" or type == "cohorts":
            distinct_cohorts = set(x.model.cohort for x in self.signatures)
            return sum(c.x.shape[1] for c in distinct_cohorts) \
                if samples else len(distinct_cohorts)
        else:
            raise ValueError("type must be one of 'signature', 'model', "
                             "'cohort'")

    @property
    def cohort_model_count(self) -> Counter:
        """Number of models from each cohort which contain member signatures."""

        return Counter((m.cohort.name, len(m.signatures)) for m in
                       set(s.model for s in self.signatures))

    def member_similarity(self,
                          utri: bool = False) -> pd.DataFrame:
        """Pairwise cosine similarity of the member signatures."""

        sig_df: pd.DataFrame = pd.concat(
            self.signatures,
            axis=1
        )
        sig_df.columns = range(sig_df.shape[1])
        sig_cs: pd.DataFrame = pd.DataFrame(
            cosine_similarity(sig_df.T),
            index=sig_df.columns,
            columns=sig_df.columns
        )
        sig_cs = sig_cs.stack().reset_index()
        sig_cs.columns = ['a', 'b', 'cosine_similarity']
        sig_cs['cohort_a'] = sig_cs['a'].apply(
            lambda x: self.signatures[x].model.cohort.name
        )
        sig_cs['cohort_b'] = sig_cs['b'].apply(
            lambda x: self.signatures[x].model.cohort.name
        )
        sig_cs['within'] = sig_cs['cohort_a'] == sig_cs['cohort_b']
        # Remove diagonal
        sig_cs = sig_cs.loc[sig_cs['a'] != sig_cs['b'], :]
        sig_cs['pair'] = sig_cs.apply(
            lambda x: set([x['a'], x['b']]),
            axis=1
        )
        if utri:
            sig_cs = sig_cs.drop_duplicates(subset=['pair'])
        return sig_cs

    def plot_member_similarity(self,
                               split_cohort: bool = True) -> plotnine.ggplot:
        sim: pd.DataFrame = self.member_similarity()
        plt: plotnine.ggplot
        if split_cohort:
            plt = (
                    plotnine.ggplot(
                        sim,
                        mapping=plotnine.aes(
                            x="cohort_a",
                            y="cosine_similarity",
                            fill="within"
                        )
                    ) + plotnine.geom_boxplot()
            )
        else:
            plt = (
                    plotnine.ggplot(
                        sim.drop_duplicates(subset=['pair']),
                        mapping=plotnine.aes(
                            y="cosine_similarity"
                        )
                    ) +
                    plotnine.geom_boxplot()
            )
        return plt

    def member_feature_weights(self,
                               top_n: int = None,
                               unit_scale: bool = True) -> pd.DataFrame:
        """Weights of features in all member signatures of this cluster.

        This is returned in long form, intended to be used for plotting using
        plotnine."""

        feature_df: pd.DataFrame = pd.concat(
            self.signatures,
            axis=1
        )
        feature_df.columns = range(feature_df.shape[1])
        # Convert each to unit vectors if requested, to control differences in
        # scale between models with different ranks
        if unit_scale:
            feature_df = feature_df / np.linalg.norm(feature_df, axis=0)
        feature_df = feature_df.loc[
                     (feature_df
                      .mean(axis="columns")
                      .sort_values(ascending=False)
                      .index), :]
        feature_df = feature_df.loc[
                     feature_df.mean(axis="columns").sort_values(ascending=False).index, :
                     ]
        if top_n is None:
            top_n = feature_df.shape[0]
        feature_df = feature_df.iloc[:min(top_n, feature_df.shape[0]), :]
        feature_df_stack: pd.DataFrame = (
            feature_df
            .stack()
            .reset_index()
        )
        feature_df_stack.columns = ['feature', 'signature', 'weight']
        feature_df_stack['feature'] = pd.Categorical(
            feature_df_stack['feature'],
            categories=reversed(feature_df.index),
            ordered=True
        )
        # Attach cohort name as columns
        feature_df_stack['cohort'] = feature_df_stack['signature'].apply(
            lambda x: self.signatures[x].model.cohort.name
        )
        return feature_df_stack

    def plot_feature_variance(
            self,
            top_n: Optional[int] = None,
            split_cohort: bool = False
    ):
        """Box plot for how much a feature weight varies."""
        # TODO: Needs scaling somehow?
        feature_df: pd.DataFrame = self.member_feature_weights(top_n=top_n)
        mapping: plotnine.aes = (
            plotnine.aes(y="weight", x="feature", fill="cohort")
            if split_cohort else
            plotnine.aes(y="weight", x="feature"))
        plt: plotnine.ggplot = (
                plotnine.ggplot(
                    feature_df,
                    mapping=mapping
                ) +
                plotnine.geom_boxplot() +
                plotnine.coord_flip()
        )
        return plt

    def plot_mds(self,
                 **kwargs) -> plotnine.ggplot:
        """Plot signatures in this cluster in a 2D space.

        Produce a 2D representation of member signatures of this cluster, with
        color indicting cohort and shape indicating rank. Can pass any arguments
        used by scikit-learn MDS constructor in kwargs.
        """
        df: pd.DataFrame = Signature.mds(self.signatures, **kwargs)
        return (
                plotnine.ggplot(df,
                                plotnine.aes(x='dim_1', y='dim_2', color='cohort',
                                             shape='factor(model_rank)')) +
                plotnine.geom_point() +
                plotnine.labs(x="", y="", shape="Rank", color="Cohort")
        )

    @staticmethod
    def as_dataframe(clusters: List[Cluster]) -> pd.DataFrame:
        """Concatenated table with the mean signatures for each cluster.

        :param clusters: Clusters to concatenate
        :returns: Concatenated DataFrame with each mean signature as a column
        """
        return pd.concat([x.mean_signature for x in clusters], axis=1)

    @staticmethod
    def cosine_similarity(clusters: List[Cluster]) -> pd.DataFrame:
        """Get the pairwise cosine similarity between clusters.

        :param clusters: Clusters to calculate similarity between
        :returns: DataFrame with pairwise cosine similarity
        """

        cat_df: pd.DataFrame = Cluster.as_dataframe(clusters)
        return cosine_similarity(cat_df.T)

    @staticmethod
    def merge(clusters: Iterable[Cluster]) -> Cluster:
        """Merge clusters.

        :param clusters: Cluster to merge.
        :returns: A new cluster containing the union of the signatures.
        """

        return Cluster(set(itertools.chain.from_iterable(
            x.signatures for x in clusters
        )))

    def __repr__(self) -> str:
        return f'Cluster[nsigs={len(self.signatures)}]'


class Combiner:

    def __init__(self,
                 cohorts: Iterable[Cohort]
                 ) -> None:
        self.__cohorts: List[Cohort] = list(cohorts)
        self.__clusters: List[Cluster] = [Cluster([y]) for y in
                                          itertools.chain.from_iterable((x.signatures for x in cohorts))
                                          ]
        self.__linear_combinations: List[
            Tuple[Signature, List[Signature]]] = []
        self.__removed_clusters: List[Cluster] = []

    @property
    def cohorts(self) -> List[Cohort]:
        return self.__cohorts

    @property
    def clusters(self) -> List[Cluster]:
        return self.__clusters

    @property
    def removed_clusters(self) -> List[Cluster]:
        return self.__removed_clusters

    def merge_similar(self,
                      cosine_threshold: float = 0.9,
                      density_threshold: float = 0.98
                      ) -> None:
        """Merge signatures which are highly similar based on cosine similarity.

        Signatures are grouped into clusters when they all share similarity
        greater than the specified threshold. A signature can end up in
        multiple clusters as a result of this grouping. Use
        refine_multimembers to force a singular membership after merging.

        :param cosine_threshold: Cosine similarity above which to consider
            samples similar.
        """

        loop_i: int = 1
        while True:
            cos_df: pd.DataFrame = Cluster.cosine_similarity(self.clusters)
            adj: np.ndarray = cos_df > cosine_threshold
            np.fill_diagonal(adj, False)
            g = nx.from_numpy_array(adj)

            cliques = list(nx.find_cliques(g))
            logging.info("Merge iteration %s: %s cliques found in %s clusters",
                         loop_i, len(cliques), len(self.clusters))

            # Terminate if no cliques |clique|>1 to merge
            if max(len(x) for x in cliques) < 2:
                break

            # Merge largest clique
            # TODO: Continue merging cliques after removing vertices which
            # have already been merged
            sorted_cliques: List[Set[int]] = list(reversed(sorted(
                cliques,
                key=len
            )))
            in_clusters: Set[int] = set()
            new_clusters: List[Cluster] = []
            for clique in sorted_cliques:
                remaining_members: Set[int] = set(clique).difference(in_clusters)
                if len(remaining_members) > 1:
                    new_cluster: Cluster = Cluster.merge(
                        [self.clusters[i] for i in remaining_members])
                    new_clusters.append(new_cluster)
                    in_clusters.update(remaining_members)
            # Remove those which were added to cliques
            # and add new clusters
            for rm in [self.clusters[i] for i in in_clusters]:
                self.__clusters.remove(rm)
            for c in new_clusters:
                self.__clusters.append(c)
            # # Merge clusters in cliques
            # self.__clusters = [
            #     Cluster.merge(
            #         [self.clusters[i] for i in clique]
            #     ) for clique in cliques
            # ]

    def identify_cohort_specific(
            self,
            cohort_proportion: float = 0.95
    ) -> List[Cluster]:
        """Find clusters which are consistent in one or more cohorts.

        A cluster may be unique to a cohort, being consistently recovered in
        data from that cohort but not in some others. These may appear to
        have poor support globally (considering all cohorts), but when
        there is strong consensus within a cohort it may be of interest to
        retain these clusters.
        """
        # Find maximum number of models for each cohort
        max_support: Dict[Tuple[str, int], int] = {}
        for cohort in self.cohorts:
            for rank, count in Counter(
                    len(m.signatures) for m in cohort.models).items():
                max_support[(cohort.name, rank)] = count
        cluster_df: pd.DataFrame = pd.concat(
            [pd.Series(c.cohort_model_count) for c in self.clusters],
            axis=1
        ).fillna(0)
        cluster_prop: pd.DataFrame = (
            (cluster_df.T / pd.Series(max_support))
            .stack().stack()
            .reset_index()
            .set_axis(['cluster_idx', 'rank', 'cohort', 'prop'], axis=1)
        )
        specific: pd.DataFrame = cluster_prop[
            cluster_prop['prop'] >= cohort_proportion]

        # For now just pull out those clusters and return
        return [self.clusters[i] for i in set(specific['cluster_idx'])]

    def remove_low_support(
            self,
            support_required: Union[float, int] = 0.2,
            support_type: Literal["signature", "model", "cohort"] = "model",
            support_samples: bool = False,
            signature_floor: int = 2,
            exempt_clusters: List[Cluster] = None,
            retain_alpha: float = 0.05,
            only_cohort_data: bool = False
    ) -> List[Cluster]:
        """Remove signatures do not appear frequently among models or cohorts.

        This looks for clusters with signatures which appear in a small number
        of models (or just a small number of signatures, or signatures
        from a small number of cohorts). Any signatures below the threshold
        for low support are retained only if removing them cause a worse
        model fit than removing the other signatures (the model fit for all
        the others pooled). This can be evaluated on the full data, or only on
        the data from the cohorts which member signatures are drawn from.

        :param support_required: Either proportion or absolute number of below
            which the signature will be considered to have low support.
        :param support_type: What to use to determine support; either the
            number of signatures, number of models cluster members appear in,
            or number of cohorts cluster members appear in.
        :param support_samples: For model or cohort, count the number of
            samples rather than each model or cohort as one. Useful when
            cohorts very uneven size.
        :param signature_floor: Remove any clusters with fewer than this
            number of member signatures.
        :param exempt_clusters: Clusters which will not be removed even if
            meeting low support criteria.
        :param retain_alpha: Signatures are retained if removing them
            has a significantly different impact on model fit compared
            to any good signature. Tested with a Kruskal-Wallis test using
            this parameter as a threshold. Set to 0 to reject all low support
            signatures.
        :param only_cohort_data: Only use data from the cohorts from which
            this cluster has support when evaluating the change in model fit
            from omitting this signature.
        """

        # Determine the maximum possible support
        support_total: int
        if support_type == 'signature' or support_type == 'signatures':
            if support_samples:
                logging.warning("support_samples not supported for signatures.")
            support_total = len(set(itertools.chain.from_iterable(
                c.signatures for c in self.clusters
            )))
        elif support_type == 'model' or support_type == 'models':
            support_total = (
                sum([x.x.shape[1] * len(x.models) for x in self.cohorts])
                if support_samples else
                sum(len(x.models) for x in self.cohorts)
            )
        elif support_type == 'cohort' or support_type == 'cohorts':
            support_total = (
                sum([x.x.shape[1] for x in self.cohorts])
                if support_samples else
                len(self.cohorts)
            )
        else:
            raise ValueError("support_type should be one of "
                             "'signature', 'model', or 'cohort'.")

        req_support: int
        match support_required:
            case int():
                req_support = support_required
            case float():
                req_support = math.ceil(support_total * support_required)
            case _:
                raise TypeError('support_required should be int or float')
        req_support = max(1, min(req_support, support_total))

        exempt_clusters = exempt_clusters if exempt_clusters is not None else []

        # Remove any clusters below the floor
        floored: List[Cluster] = [
            y for y in self.clusters if len(y.signatures) < signature_floor]
        for y in floored:
            self.clusters.remove(y)
            self.__removed_clusters.append(y)

        # Identify remaining low support clusters
        low_support_clusters: List[Cluster] = [
            x for x in self.clusters
            if x.support(type=support_type, samples=support_samples)
               < req_support
               and
               x not in exempt_clusters
        ]
        other_clusters: List[Cluster] = [
            x for x in self.clusters if x not in low_support_clusters
        ]

        # Does removing this signature have a bigger impact than removing
        # any of the non-low support signatures?
        good_df: pd.DataFrame = Cluster.as_dataframe(other_clusters)
        for i_low in low_support_clusters:
            # If the alpha is 0 or lower, will never retain so skip calculation
            if retain_alpha <= 0:
                self.clusters.remove(i_low)
                self.__removed_clusters.append(i_low)
                continue
            i_sigs: pd.DataFrame = pd.concat(
                [Cluster.as_dataframe([i_low]), good_df],
                axis=1,
            )
            # Rename to avoid undesired behaviour with dropping column by index
            i_sigs.columns = [f's{i}' for i in range(i_sigs.shape[1])]
            # Get the input data to evaluate model fit on
            x: pd.DataFrame = (
                i_low.member_data if only_cohort_data else self.cohort_data
            )

            # Find model fit for all samples with each signature removed
            loo_modelfit: List[pd.Series] = [
                _reapply_model(y=x,
                               w=i_sigs.drop(columns=[i]),
                               colors=None,
                               input_validation=lambda x: x,
                               feature_match=match_identical).model_fit
                for i in i_sigs.columns
            ]

            # Compare the low support signature model fit to the others
            concat_mf: pd.Series = pd.concat(loo_modelfit[1:])
            h, p = kruskal(loo_modelfit[0],
                           concat_mf)
            # Only retain is the model fit is worse when removing this
            # signature than the others, and this is significantly different
            if not (loo_modelfit[0].mean() < concat_mf.mean() and
                    p <= retain_alpha):
                self.clusters.remove(i_low)
                self.__removed_clusters.append(i_low)

    @property
    def cohort_data(self) -> pd.DataFrame:
        """Combined data for all cohorts."""

        merged_df: pd.DataFrame = pd.concat([x.x for x in self.cohorts],
                                            axis=1)
        # If there are duplicate columns removed them but emit warning
        orig_shape: Tuple[int, int] = merged_df.shape
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        if orig_shape != merged_df.shape:
            logging.warning("Sample names duplicated between cohorts. If these "
                            "are distinct samples give them different names, "
                            "otherwise they will be treated as one sample "
                            "during signature combining.")
        return merged_df

    def remove_linear_combinations(
            self,
            cosine_threshold: float = 0.9,
            min_small_support_ratio: float = 0.5,
            support_type: Literal["signature", "model", "cohort"] = "model",
            support_samples: bool = False,
            do_removal: bool = True
    ) -> None:
        """Remove signatures which can be expressed as linear combinations.

        Some signatures might be a combination of multiple others, which
        commonly co-occur in some of the cohorts. We can remove these and
        keep only the constituent signatures. However, do not want to discard
        well supported large signatures as they are combination of smaller but
        less well supported signatures.

        :param cosine_threshold: Model fit threshold
        :param do_removal: Remove the signatures which are combinations. Set to
            False to return the identified signatures without removing.
        """

        removed: List[Tuple[Cluster, List[Cluster]]] = []
        for cluster in self.clusters:
            # Reject any cluster whose support is not sufficient
            c_support: int = cluster.support(type=support_type,
                                             samples=support_samples)
            accept_clusters: List[Cluster] = [
                c for c in self.clusters
                if (c is not cluster and
                    (c.support(type=support_type, samples=support_samples) /
                     c_support) >= min_small_support_ratio
                    )
            ]
            # If one or zero clusters acceptable, will never be usefully
            # describe as a mix so skip further tests
            if len(accept_clusters) < 2:
                continue

            # Make a matrix of all the other signatures
            # and perform NNLS to describe candidate signature as a mixture
            w: pd.DataFrame = Cluster.as_dataframe(accept_clusters)
            signature_mix: Decomposition = _reapply_model(
                y=cluster.mean_signature.to_frame(),
                w=w,
                input_validation=lambda x: x,
                feature_match=match_identical,
                colors=None
            )

            if signature_mix.model_fit.iloc[0] < cosine_threshold:
                continue
            # Take any signatures which are required to describe 99% of the
            # total weight as those which make up the one to be removed
            repr: pd.Series = signature_mix.representative_signatures(
                threshold=0.99).iloc[:, 0]
            repr_clusters: List[Cluster] = list(itertools.compress(
                accept_clusters,
                list(repr)
            ))
            removed.append((cluster, repr_clusters))

        if do_removal:
            for cluster, components in removed:
                self.clusters.remove(cluster)
                self.__linear_combinations.append((cluster, components))

        return removed

    def remove_linear_combinations_2(
            self,
            cosine_threshold: float = 0.9,
            min_small_support_ratio: float = 0.5,
            support_type: Literal["signature", "model", "cohort"] = "model",
            support_samples: bool = False,
            do_removal: bool = True
    ):
        """Linear combination removal with removal of smaller poorly supported
        signatures along with larger ones."""

        to_test: List[Cluster] = sorted(
            self.clusters,
            key=lambda x: x.support(type=support_type,
                                    samples=support_samples)
        )
        removed: Set[Cluster] = set()

        while len(to_test) > 0:
            # Start from highest support cluster
            cluster: Cluster = to_test[-1]
            remaining_clusters: List[Cluster] = [
                x for x in self.clusters
                if x not in removed and x is not cluster
            ]
            c_support: int = cluster.support(
                type=support_type, samples=support_samples
            )
            threshold_met: List[bool] = [
                (c.support(type=support_type, samples=support_samples) /
                 c_support) > min_small_support_ratio
                for c in remaining_clusters
            ]

            # Make a matrix of all other remaining signatures and performing
            # NNLS to describe candidate signature as a mixture
            w: pd.DataFrame = Cluster.as_dataframe(remaining_clusters)
            signature_mix: Decomposition = _reapply_model(
                y=cluster.mean_signature.to_frame(),
                w=w,
                input_validation=lambda x: x,
                feature_match=match_identical,
                colors=None
            )

            if signature_mix.model_fit.iloc[0] < cosine_threshold:
                # This is not a mixture of any other signatures
                to_test.remove(cluster)
                continue

            # This is a mixture of some of the other signatures
            repr: pd.Series = signature_mix.representative_signatures(
                threshold=0.98).iloc[:, 0]
            repr_clusters: List[Cluster] = list(itertools.compress(
                remaining_clusters,
                list(repr)
            ))
            repr_threshold: List[bool] = list(itertools.compress(
                threshold_met,
                list(repr)
            ))
            # If all the signatures which contribute to this one fall above
            # the relative support threshold, remove this signature
            if all(repr_threshold):
                removed.update({cluster})
                self.__linear_combinations.append(
                    (cluster, repr_clusters)
                )
                to_test.remove(cluster)
            # Remove any signatures which fall below support threshold,
            # but do not remove this signature form the candidate pool
            else:
                for below in itertools.compress(repr_clusters,
                                                (not x for x in repr_threshold)):
                    removed.update({below})
                    if below in to_test:
                        to_test.remove(below)

        if do_removal:
            for x in removed:
                self.clusters.remove(x)

        return removed

    def plot_feature_variance(
            self,
            split_cohort: bool = True,
            top_n: Optional[int] = 20,
            label_fn: Callable[[List[str]], List[str]] = None,
            unit_scale: bool = True
    ) -> plotnine.ggplot:
        """Plot feature variance for all signatures."""
        feature_df: pd.DataFrame = pd.concat([
            (x.member_feature_weights(top_n=top_n)
             .assign(cluster=x.label))
            for i, x in enumerate(self.clusters)
        ])
        mapping: Dict[str, str] = dict(
            y="weight",
            x="feature"
        )
        if split_cohort:
            mapping['fill'] = "cohort"
        if label_fn is not None:
            feature_df['feature'] = label_fn(feature_df['feature'])
        plt: plotnine.ggplot = (
                plotnine.ggplot(
                    feature_df,
                    mapping=plotnine.aes(**mapping)
                ) +
                plotnine.geom_boxplot(outlier_size=0.2) +
                plotnine.coord_flip()
        )
        if split_cohort:
            plt = plt + plotnine.facet_wrap("cluster",
                                            scales="free_y",
                                            ncol=len(self.clusters))
        else:
            plt = plt + plotnine.facet_wrap(facets="cluster", scales="free_y")
        return plt

    def plot_member_similarity(self,
                               split_cohort: bool = True):
        sim: pd.DataFrame = pd.concat([
            (x.member_similarity()
             .assign(cluster=x.label))
            for i, x in enumerate(self.clusters)
        ])
        plt: plotnine.ggplot
        if split_cohort:
            plt = (
                    plotnine.ggplot(
                        sim,
                        mapping=plotnine.aes(
                            x="cohort_a",
                            y="cosine_similarity",
                            fill="within"
                        )
                    ) +
                    plotnine.geom_boxplot() +
                    plotnine.facet_wrap("cluster") +
                    plotnine.labs(x="Cohort", y="Cosine Similarity")
            )
        else:
            plt = (
                    plotnine.ggplot(
                        sim.drop_duplicates(subset=['pair']),
                        mapping=plotnine.aes(
                            x="cluster",
                            y="cosine_similarity"
                        )
                    ) +
                    plotnine.geom_boxplot() +
                    plotnine.labs(x="Cluster", y="Cosine Similarity")
            )
        return plt

    def plot_mds(
            self,
            hull: bool = True,
            **kwargs
    ) -> plotnine.ggplot:
        """Plot all clusters member signatures in reduced dimensions."""

        sig_list: List[Signature] = list(itertools.chain.from_iterable(
            x.signatures for x in self.clusters))
        signatures: pd.DataFrame = pd.concat(
            sig_list + [x.mean_signature for x in self.clusters], axis=1)
        df: pd.DataFrame = Signature.mds(signatures, **kwargs)

        # Attach metadata
        df = df.assign(cohort="Mean", model_rank=0, cluster=None)
        num_centroids: int = len(self.clusters)
        # For non-centroid signatures
        df['cohort'] = [
            x.model.cohort.name for x in sig_list] + (["Mean"] * num_centroids)
        df['model_rank'] = [
            len(x.model.signatures) for x in sig_list] + ([0] * num_centroids)
        df['cluster'] = list(
            itertools.chain.from_iterable(
                [x.label] * len(x.signatures) for x in self.clusters
            )) + list(x.label for x in self.clusters)
        # Label if is a mean signature
        df['is_mean'] = (
                np.array(range(df.shape[0])) < (df.shape[0] - num_centroids))
        plt: plotnine.ggplot = (
                plotnine.ggplot(
                    df,
                    plotnine.aes(x="dim_1", y="dim_2", color="cohort",
                                 shape="factor(model_rank)")) +
                plotnine.geom_point(size=0.5) +

                plotnine.geom_text(
                    plotnine.aes(label="cluster"),
                    data=df[~df['is_mean']],
                    size=6,
                    color="black"
                )
        )
        if hull:
            plt = plt + plotnine.stat_hull(
                plotnine.aes(group="cluster"), color="black")
        return plt

    def label_by_match(self,
                       external_model: Comparable) -> pd.DataFrame:
        """Label clusters by their similarity to an existing model."""
        match: pd.DataFrame = match_signatures(
            Cluster.as_dataframe(self.clusters),
            external_model
        )
        for match_row in match.loc[~match['a'].isna(), :].iterrows():
            match_row = match_row[1]
            cluster_idx: Optional[int] = match_row['a']
            other_lbl: Optional[str] = match_row['b']
            other_cos: Optional[float] = match_row['pair_cosine']
            if other_lbl is not None:
                self.clusters[cluster_idx].label = (
                    f'{other_lbl} ({other_cos:.2f})')
            else:
                self.clusters[cluster_idx].label = (
                    f'Cluster {cluster_idx}'
                )
        return match


def example_cohort_structure() -> Dict[str, pd.DataFrame]:
    """Make 5 cohorts of data with a complex structure.

    This set of cohorts is suitable to evaluate whether we are capturing the
    different kind of situations we expect during signature combining. These
    are:
    * highly similar signatures which should be merged
    * signatures which are linear combinations of others which should be removed
    * low support signatures (present in few cohorts) which can be
    *   *   uninformative, in which case removed
    *   *   informative cohort specific, in which case retained

    To this end, we make 5 cohorts based on the 5 ES model signatures.
    1.  Contains all 5 ES
    2.  Doesn't contain ES_Bifi
    3.  Contains an extra signature IS_1 which is informative
    4.  Completely shuffled, does not fit at all
    5.  Doesn't contain ES_Bact or ES_Esch
    6.  Merged ES_Firm and ES_Prev

    Each is returned as a named tuple containing name, h, w, x
    """
    from cvanmf import models

    CohortTuple = namedtuple("CohortTuple", "name w h x")
    # C1 - All ES
    es_w: pd.DataFrame = models.five_es().w
    c1_w: pd.DataFrame = es_w.copy()
    c1_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c1_w.shape[1], 100)),
        index=c1_w.columns
    )
    c1_h: pd.DataFrame = c1_h / c1_h.sum()
    c1_x: pd.DataFrame = c1_w.dot(c1_h)
    c1 = CohortTuple("All", c1_w, c1_h, c1_x / c1_x.sum())

    # C2 - No ES_Bifi
    c2_w: pd.DataFrame = es_w.drop(columns=['ES_Bifi'])
    c2_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c2_w.shape[1], 200)),
        index=c2_w.columns
    )
    c2_h: pd.DataFrame = c2_h / c2_h.sum()
    c2_x: pd.DataFrame = c2_w.dot(c2_h)
    c2 = CohortTuple("No_Bifi", c2_w, c2_h, c2_x / c2_x.sum())

    # C3 - Extra signature
    c3_w: pd.DataFrame = es_w.copy()
    c3_w.loc[:, 'IS_1'] = es_w['ES_Firm'].sample(frac=1).values
    c3_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c3_w.shape[1], 150)),
        index=c3_w.columns
    )
    c3_h: pd.DataFrame = c3_h / c3_h.sum()
    c3_x: pd.DataFrame = c3_w.dot(c3_h)
    c3 = CohortTuple("IS_1", c3_w, c3_h, c3_x / c3_x.sum())

    # C4 - Randomly shuffled, which should make irrelevant signatures
    c4_w: pd.DataFrame = es_w.copy()
    c4_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c4_w.shape[1], 100)),
        index=c4_w.columns
    )
    c4_h: pd.DataFrame = c4_h / c4_h.sum()
    c4_x: pd.DataFrame = c4_w.dot(c4_h)
    for i in range(c4_x.shape[1]):
        c4_x.iloc[:, i] = c4_x.iloc[:, i].sample(frac=1)
    c4 = CohortTuple("Random", c4_w, c4_h, c4_w / c4_w.sum())

    # C5 - No Bifi or Esch
    c5_w: pd.DataFrame = es_w.drop(columns=['ES_Bact', 'ES_Esch'])
    c5_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c5_w.shape[1], 200)),
        index=c5_w.columns
    )
    c5_h: pd.DataFrame = c5_h / c5_h.sum()
    c5_x: pd.DataFrame = c5_w.dot(c5_h)
    c5 = CohortTuple("No_Bifi_Esch", c5_w, c5_h, c5_x / c5_x.sum())

    # C6 - Bact and Prev merged
    c6_w: pd.DataFrame = es_w.copy()
    c6_w['ES_Prev+Bact'] = c6_w['ES_Prev'] + c6_w['ES_Bact']
    c6_w = c6_w.drop(columns=['ES_Bact', 'ES_Prev'])
    c6_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c6_w.shape[1], 200)),
        index=c6_w.columns
    )
    c6_h: pd.DataFrame = c6_h / c6_h.sum()
    c6_x: pd.DataFrame = c6_w.dot(c6_h)
    c6 = CohortTuple("Merge_Prev_Bact", c6_w, c6_h, c6_x / c6_x.sum())

    return {x.name: x for x in [c1, c2, c3, c4, c5, c6]}


def split_dataframe_to_cohorts(
        x: pd.DataFrame,
        cohort_labels: pd.Series,
        min_size: int = 0
) -> Dict[Any, pd.DataFrame]:
    """Split a DataFrame into multiple, based on the provided cohort labels."""

    cohort_count: pd.Series = cohort_labels.groupby(cohort_labels).count()
    cohort_include: pd.Series = cohort_count.loc[cohort_count >= min_size]
    labels: pd.Series = cohort_labels.loc[
        cohort_labels.isin(cohort_include.index)]
    return {
        lbl: x.loc[:, cohort_labels.loc[cohort_labels == lbl].index]
        for lbl in cohort_include.index
    }

