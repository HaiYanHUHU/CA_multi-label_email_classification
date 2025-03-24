import typing as t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

from utils import ObjectEncoder


class Dataset:

    def __init__(self, df: pd.DataFrame, x: t.Union[t.Iterable[str], str], y: str = 'y', split: float = 0.2):
        """

        :param df: pd.DataFrame source of all data
        :param x: str|Iterable[str] representing a column in the dataframe used as independent variable
        :param y: str representing a column in the dataframe used as a dependent variable
        :param split: float percentage of data to perform a test split
        """

        self.df = df.reset_index(drop=True).copy()
        self.x = (x,) if isinstance(x, str) else x

        assert len(df[df[list(self.x)].isna().any(axis=1)]) == 0, "nan values in x columns"

        self.y = y
        self.split_ = split
        self.vectorisers = {}
        self.obj_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def tuplefy(self, cols: tuple[str]) -> None:
        """
        Tuplefy
        :param cols:
        :return:
        """
        oe = ObjectEncoder()
        tuples = list(zip(*(self.df[col] for col in cols)))
        self.df['y'] = oe.fit_transform(tuples)
        self.obj_encoders[cols] = oe

    def detuplefy(self, orig: tuple[str], encoded: int) -> tuple:
        """
        Detuplefy
        :param orig:
        :param encoded:
        :return:
        """
        return self.obj_encoders[orig].classes_[encoded]

    def vectorise(self, col: str, max_features: int = 1000) -> None:
        """
        vectorises text data, adds to X
        :param col: str column in the df to vectorise (the series being refered should be non-nan)
        :param max_features: int max features
        :return: None
        """
        vec = TfidfVectorizer(max_features=max_features)
        tmp = vec.fit_transform(self.df[col])
        x_cols = [f"{col}_{w}" for w in vec.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tmp.toarray(), columns=x_cols)

        l = list(self.x)
        if col in self.x:
            l.remove(col)

        l.extend(x_cols)

        assert len(tfidf_df[tfidf_df.isna().any(axis=1)]) == 0, "tfid error"

        self.x = tuple(l)

        ll = len(self.df)
        assert ll == len(tfidf_df)

        self.df = pd.concat([self.df, tfidf_df], axis=1)

        assert len(self.df) == ll, "index misalignment"

        self.vectorisers[col] = vec

    def devectorise(self, col: str, encoded: t.Sequence[int]) -> str:
        """
        devectorises
        :param col: str original name of the column
        :param tfid: array with vectorised data
        :return: str decoded string
        """
        return ' '.join(self.vectorisers[col].inverse_transform([encoded]))

    def is_ready(self) -> bool:
        """
        checks if the split has been done and sanity checks
        :return: boolean indicates if the dataset is ready for training/evaluating
        """
        return self.X_test is not None and 'y' in self.df.columns

    def split(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[list(iter(self.x))], self.df['y'],
                                                                                test_size=self.split_)