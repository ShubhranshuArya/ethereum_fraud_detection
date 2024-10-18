from zenml import pipeline, Model
from steps.data_ingestion_step import data_ingestion_step
from steps.data_modelling_step import data_modelling_step
from steps.data_splitting_step import data_splitter_step
from steps.feature_cleaning_step import feature_cleaning_step
from steps.feature_engineering_step import feature_engineering_step
from steps.missing_value_handling_step import missing_value_handling_step


@pipeline(
    model=Model(
        # The name uniquely identifies the model.
        name="ethereum_fraud_detection"
    )
)
def ml_pipeline():
    """Complete End-To-End Pipeline"""

    # Data Ingestion Step
    raw_df = data_ingestion_step(
        file_path="/Users/botcoder/MyDocs/AiLearn/Antern Learn/project/ethereum-fraud-detection/data/ethereum_fraud_labeled_data.csv",
    )

    # Handle Missing Values Step
    missing_values_handled_df = missing_value_handling_step(
        df=raw_df,
        strategy="drop",
    )

    # Feature Cleaning Step
    feature_cleaned_df = feature_cleaning_step(
        df=missing_values_handled_df,
        target_column="FLAG",
        unwanted_feature_list=["Unnamed: 0", "Index"],
    )

    # Feature Engineering Step
    feature_engineered_df = feature_engineering_step(
        df=feature_cleaned_df,
        target_column="flag",
    )

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(
        feature_engineered_df,
        target_column="flag",
    )

    # Data Modelling Step
    model = data_modelling_step(
        X_train=X_train,
        y_train=y_train,
    )

    # TODO: Implement model evaluation

    return model
