import pandas as pd
import numpy as np
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from pandas.api.types import (
    is_string_dtype, is_numeric_dtype,
    is_datetime64_any_dtype, is_bool_dtype, is_timedelta64_dtype
)


class Clean_File:

    def __init__(self, file_path, log_path="imputer_report.log"):
        self.file_path = file_path
        self.data = self.load_file()

        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"  
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Started processing file: {self.file_path}")

    def load_file(self):
        try:
            if isinstance(self.file_path, str):
                if self.file_path.endswith(".csv"):
                    df = pd.read_csv(self.file_path)
                elif self.file_path.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(self.file_path)
                else:
                    raise ValueError("Unsupported file format. Use only CSV or Excel.")
            elif hasattr(self.file_path, "name"):
                if self.file_path.name.endswith(".csv"):
                    df = pd.read_csv(self.file_path)
                elif self.file_path.name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(self.file_path)
                else:
                    raise ValueError("Unsupported file format. Use only CSV or Excel.")
            else:
                raise ValueError("Invalid file input.")

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            return df

        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def data_information(self):
        columns = self.data.columns
        col_wise_null = self.data.isnull().sum()
        total_null = self.data.isnull().sum().sum()
        total_count = len(self.data)
        not_null = self.data.notnull().sum()

        summary = pd.DataFrame({
            "Missing": col_wise_null,
            "Not Missing": not_null
        })

        self.logger.info("Data information extracted.")
        return columns, col_wise_null, total_null, total_count, not_null, summary
    

    def smart_imputer(self):
        self.data_dummy = None
        self.data_dummy = self.data.copy(deep=True)
        report = []
        drop_cols = []

        for col in self.data_dummy:
            missing_pct = self.data_dummy[col].isnull().mean() * 100
            dtype = self.data_dummy[col].dtype

            if missing_pct > 40:
                drop_cols.append(col)
                msg = f"[{col}] -> Missing {missing_pct:.2f}% (>40%), column dropped."
                self.data_dummy.drop(col, axis=1, inplace=True)
                report.append(msg)
                self.logger.info(msg)
                continue

            if is_string_dtype(self.data_dummy[col]) or is_bool_dtype(self.data_dummy[col]):
                n_unique = self.data_dummy[col].nunique(dropna=True)

                if is_bool_dtype(self.data_dummy[col]):
                    mode_val = self.data_dummy[col].mode(dropna=True)[0]
                    self.data_dummy[col].fillna(mode_val, inplace=True)
                    msg = f"[{col}] -> Boolean, imputed with mode."
                    report.append(msg)
                    self.logger.info(msg)

                elif n_unique < 10:
                    mode_val = self.data_dummy[col].mode(dropna=True)[0]
                    self.data_dummy[col].fillna(mode_val, inplace=True)
                    msg = f"[{col}] -> Small categorical, imputed with Mode."
                    report.append(msg)
                    self.logger.info(msg)

                else:
                    self.data_dummy[col] = self.data_dummy[col].fillna(method="ffill").fillna(method="bfill")
                    msg = f"[{col}] -> Multi-category text, imputed with ffill/bfill."
                    report.append(msg)
                    self.logger.info(msg)

            elif is_numeric_dtype(self.data_dummy[col]):
                if missing_pct == 0:
                    msg = f"[{col}] -> No missing values, unchanged."
                    report.append(msg)
                    self.logger.info(msg)
                    continue

                if missing_pct < 10:
                    self.data_dummy[col].fillna(self.data_dummy[col].mean(), inplace=True)
                    msg = f"[{col}] -> Missing {missing_pct:.2f}%, imputed with Mean."
                    report.append(msg)
                    self.logger.info(msg)

                elif missing_pct < 20:
                    imputer = KNNImputer(n_neighbors=5)
                    self.data_dummy[[col]] = imputer.fit_transform(self.data_dummy[[col]])
                    msg = f"[{col}] -> Missing {missing_pct:.2f}%, imputed with KNN."
                    report.append(msg)
                    self.logger.info(msg)

                else:
                    imputer = IterativeImputer(random_state=42)
                    self.data_dummy[[col]] = imputer.fit_transform(self.data_dummy[[col]])
                    msg = f"[{col}] -> Missing {missing_pct:.2f}%, imputed with Iterative (MICE)."
                    report.append(msg)
                    self.logger.info(msg)

            elif is_datetime64_any_dtype(self.data_dummy[col]):
                self.data_dummy[col] = self.data_dummy[col].fillna(method="ffill").fillna(method="bfill")
                msg = f"[{col}] -> Datetime, imputed with ffill/bfill."
                report.append(msg)
                self.logger.info(msg)

            elif is_timedelta64_dtype(self.data_dummy[col]):
                self.data_dummy[col].fillna(self.data_dummy[col].median(), inplace=True)
                msg = f"[{col}] -> Timedelta, imputed with Median."
                report.append(msg)
                self.logger.info(msg)

            else:
                msg = f"[{col}] -> Unknown dtype {dtype}, skipped."
                report.append(msg)
                self.logger.warning(msg)

        self.logger.info("Smart imputation completed.")
        return self.data_dummy, report
    
    def remove_outliers(self, method="IQR", factor=1.5):
            """
                Removes outliers from numeric columns using IQR or Z-score.
                Default: IQR with factor=1.5
        """
            df = self.data_dummy.copy()
            report = []

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if method == "IQR":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - factor * IQR
                    upper = Q3 + factor * IQR
                    before = df.shape[0]
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                    after = df.shape[0]
                    removed = before - after
                    report.append(f"[{col}] -> {removed} outliers removed using IQR.")
                    self.logger.info(report[-1])

                elif method == "zscore":
                    mean, std = df[col].mean(), df[col].std()
                    before = df.shape[0]
                    df = df[(np.abs((df[col] - mean) / std) <= factor)]
                    after = df.shape[0]
                    removed = before - after
                    report.append(f"[{col}] -> {removed} outliers removed using Z-score.")
                    self.logger.info(report[-1])

            self.data_dummy = df  
            return self.data_dummy, report

#cf = Clean_File("data.csv")
#x ,y =cf.smart_imputer()
#print(x.head())

#print(y)
#print(cf.data_information())