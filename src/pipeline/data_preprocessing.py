from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """Pipeline for imputing, encoding, scaling, and preprocessing loan application data."""

    DEFAULT_NUMERICAL = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    DEFAULT_CATEGORICAL = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_col: str = "Loan_Status",
        id_col: str = "Loan_ID",
    ):
        self.numerical_cols = numerical_cols or self.DEFAULT_NUMERICAL
        self.categorical_cols = categorical_cols or self.DEFAULT_CATEGORICAL
        self.target_col = target_col
        self.id_col = id_col

        self.num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        self.cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.transformer: Optional[ColumnTransformer] = None
        self.is_fitted: bool = False

    def _build_column_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        # Filter available columns from DataFrame
        num_cols = [c for c in self.numerical_cols if c in df.columns]
        cat_cols = [c for c in self.categorical_cols if c in df.columns]

        return ColumnTransformer(
            transformers=[
                ("num", self.num_pipeline, num_cols),
                ("cat", self.cat_pipeline, cat_cols),
            ],
            remainder="drop",
        )

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit preprocessing transformers on the dataset."""
        feature_df = df.drop(columns=[c for c in [self.id_col, self.target_col] if c in df.columns])
        self.transformer = self._build_column_transformer(feature_df)
        self.transformer.fit(feature_df)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset using fitted preprocessor pipeline."""
        if not self.is_fitted or self.transformer is None:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform().")

        feature_df = df.drop(columns=[c for c in [self.id_col, self.target_col] if c in df.columns])
        array = self.transformer.transform(feature_df)

        feature_names = self.transformer.get_feature_names_out()
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

        processed_df = pd.DataFrame(array, columns=clean_names, index=df.index)

        # Re-attach target if present in input DataFrame
        if self.target_col in df.columns:
            target_series = df[self.target_col].copy()
            mapped = target_series.astype(str).str.strip().map({"Y": 1, "N": 0, "1": 1, "0": 0})
            processed_df[self.target_col] = mapped.fillna(target_series)

        return processed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform dataset in one step."""
        return self.fit(df).transform(df)

    def save_preprocessor(self, filepath: Union[str, Path]) -> Path:
        """Serialize preprocessor instance to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load_preprocessor(cls, filepath: Union[str, Path]) -> "DataPreprocessor":
        """Load serialized preprocessor instance from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {path}")
        return joblib.load(path)


def save_processed_data(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    target_col: str = "Loan_Status",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split processed dataset into train/test sets and save to output directory.

    Args:
        df: Preprocessed DataFrame containing target column.
        output_dir: Path to directory where train.parquet and test.parquet will be saved.
        target_col: Target variable column name.
        test_size: Proportion of test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if target_col in df.columns:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[target_col]
        )
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_df.to_parquet(out_path / "train.parquet", index=False)
    test_df.to_parquet(out_path / "test.parquet", index=False)

    return train_df, test_df
