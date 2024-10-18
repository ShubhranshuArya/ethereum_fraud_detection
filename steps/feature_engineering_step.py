from zenml import step
import pandas as pd

from src.feature_engineering import (
    FeatureEngineeringHandler,
    NormalizeFeatureEngineeringStrategy,
)


@step
def feature_engineering_step(
    df: pd.DataFrame,
    target_column: str,
    features: list = [],
) -> pd.DataFrame:
    """
    Applies feature engineering transformations to the input DataFrame as a ZenML Step.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing features to be engineered.

    Returns:
        pd.DataFrame: The DataFrame with features engineered and normalized.
    """

    feature_engineering_handler = FeatureEngineeringHandler(
        NormalizeFeatureEngineeringStrategy(features)
    )
    feature_engineered = feature_engineering_handler.apply_transformation(
        df, target_column
    )
    return feature_engineered
