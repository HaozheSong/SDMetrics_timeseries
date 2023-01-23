import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class FeatureDistSimilarity(TimeSeriesMetric):
    """We compute the marginal distribution of values for a single fixed feature f. For example, for the feature key “latitude”, this would compute a distribution over all the latitudes seen in every time series in the dataset. We compare this distribution between synthetic and real data. 

    Fix a set of features F={f1,...,fm} whose joint distribution you want to compare. (This can be used to evaluate joint distributions or marginal distributions. If you only want to look at marginals (ie one feature at a time) just let F contain a single feature)."""

    name = "Feature distributional similarity"
    goal = Goal.MINIMIZE

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, target=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        attribute_cols = metadata['entity_columns'] + \
            metadata['context_columns']
        feature_cols = list(set(real_data.columns) - set(attribute_cols))

        scores = {}
        # From doc, datetime is also a feature and should be added to calculate the
        # distributional similarity. I cast the timestamp into unix integer
        # with unit (s)
        # TODO: multi-feature distribution
        for column_name in feature_cols:
            if column_name == metadata["sequence_index"]:
                real_data[column_name] = pd.to_datetime(
                    real_data[column_name]).astype(int) / 10**9
                synthetic_data[column_name] = pd.to_datetime(
                    synthetic_data[column_name]).astype(int) / 10**9
            real_column = real_data[column_name].to_numpy().reshape(-1, 1)
            synthetic_column = synthetic_data[column_name].to_numpy(
            ).reshape(-1, 1)

            if column_name in metadata['fields']:
                if metadata['fields'][column_name]['type'] in ['categorical']:
                    scores[column_name] = distribution_similarity(
                        real_data=real_column,
                        synthetic_data=synthetic_column,
                        column_names=[column_name],
                        data_type=['categorical'],
                        comparison_type='both',
                        categorical_mapping=True
                    )
                elif metadata['fields'][column_name]['type'] in ['numerical']:
                    scores[column_name] = distribution_similarity(
                        real_data=real_column,
                        synthetic_data=synthetic_column,
                        column_names=[column_name],
                        data_type=['numerical'],
                        comparison_type='both',
                        categorical_mapping=True
                    )
                elif column_name == metadata["sequence_index"]:
                    scores[column_name] = distribution_similarity(
                        real_data=real_column,
                        synthetic_data=synthetic_column,
                        column_names=[column_name],
                        data_type=['numerical'],
                        comparison_type='both',
                        categorical_mapping=True
                    )

        return scores
