import pandas as pd
import numpy as np
from iisignature import sig

class Levy():
    """
    Class for calculating Levy lead-lag scores based on a price panel of assets.
    """

    def __init__(self, price_panel: pd.DataFrame):
        """
        Initializes the Levy with a price panel of assets, calculating the returns and standardizing
        them as a preprocessing step.

        Parameters:
        - price_panel (pd.DataFrame): DataFrame containing price data for assets.
        """
        if not isinstance(price_panel, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")

        self.data = self._preprocess_data(price_panel)
        

    def _preprocess_data(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input data by calculating returns and standardizing them.

        Parameters:
        - price_panel (pd.DataFrame): DataFrame containing price data for assets.

        Returns:
        - pd.DataFrame: Processed DataFrame with returns standardized.
        """
        returns = price_panel.pct_change().dropna()
        standardized_returns = (returns - returns.mean()) / returns.std()
        return standardized_returns
    

    def calc_levy_area(self, path: np.array) -> float:
        """
        Calculates Levy area based on the provided path.

        Parameters:
        - path (np.array): Array representing the path.

        Returns:
        - float: Levy area.
        """
        path_sig = sig(path, 2) 
        levy_area = 0.5 * (path_sig[3] - path_sig[4])
        return levy_area
    

    def generate_levy_matrix(self) -> pd.DataFrame:
        """
        Generates Levy lead-lag scoring matrix for the given price panel.

        Returns:
        - pd.DataFrame: Levy lead-lag scoring matrix.
        """
        assets = self.data.columns
        n = len(assets)
        levy_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                pair_path = self.data[[assets[i], assets[j]]].values
                val = self.calc_levy_area(pair_path)
                levy_matrix[i, j] = val
                levy_matrix[j, i] = -val

        levy_matrix_df = pd.DataFrame(levy_matrix, index=assets, columns=assets)
        return levy_matrix_df
    

    def score_assets(self) -> pd.DataFrame:
        """
        Calculates the mean of each row in the Levy matrix as the corresponding asset's score.

        Returns:
        - pd.DataFrame: DataFrame with asset scores.
        """
        levy_matrix = self.generate_levy_matrix()
        lead_lag_score = levy_matrix.mean(axis=1)
        lead_lag_df = pd.DataFrame({self.data.index[-1]: lead_lag_score})
        return lead_lag_df.T