import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from levy import Levy
from hermitian import Hermitian

from typing import Tuple

class LeadLagPortfolio():
    """
    A class for managing lead-lag portfolio analysis.

    Attributes:
    - price_panel (pd.DataFrame): DataFrame containing historical prices.
    - s (dict): Dictionary to store "S" matrices.
    - g (dict): Dictionary to store "G" directed graphs.
    - dt_cluster_dict (dict): Dictionary to store cluster information.
    - selection_pct (float): Percentage for asset selection.
    - global_ranking (pd.DataFrame): Global ranking of assets.
    - global_lfs (pd.Series): Global lead-lag factor series.
    - clustered_lfs (pd.Series): Clustered lead-lag factor series.
    - return_panel (pd.DataFrame): DataFrame containing returns.
    - gp_stats (pd.DataFrame): Global portfolio statistics.
    - cp_stats (pd.DataFrame): Clustered portfolio statistics.
    """
    def __init__(self, price_panel:pd.DataFrame):
        
        self.price_panel = price_panel   
        self.return_panel = pd.DataFrame()     
        
        # Initialize S matrix and directed network dicts
        self.s = {}  
        self.g = {}  
        
        # Initialize global portfolio items
        self.global_scores = pd.DataFrame()
        self.gp_leaders_followers = pd.Series()
        self.gp_data = pd.DataFrame()
        
        # Initialize clustered portfolio items
        self.dt_cluster_dict = {}
        self.cp_leaders_followers = {}
        self.cp_data = pd.DataFrame()
        self.gcp_data = pd.DataFrame()
        
    
    def generate_matrices_and_networks(self, window_size: int = 30, min_assets: int = 40, show_progress: bool = True):
        """
        Generates lead-lag score matrices and relevant directed network for each index in the price panel 
        and stores each matrix and directed network in a dictionary with the index value as key.

        Parameters:
        - window_size (int): Size of the rolling window.
        - min_assets (int): Minimum number of assets required in the window.
        - show_progress (bool): Flag to show the progress bar.

        Note: Make sure to have the 'tqdm' library installed for the progress bar to work.
        """
        # Run a rolling window and generate lead-lag score matrices based on levy area
        for i in tqdm(range(1, len(self.price_panel) - window_size), 
                      desc='Generating Lead-Lag Scoring Matrices and Directed Networks', 
                      disable=not show_progress):
            
            # Slice the rolling window
            window_df = self.price_panel.iloc[i - 1:i + window_size, :]

            # Drop assets with missing data inside the window
            window_df = window_df.dropna(axis=1)

            # If more than min_assets are in the window, then continue
            if window_df.shape[1] >= min_assets:
                
                # Find the last index of the window as the key
                window_index = window_df.index[-1]

                # Generate Levy matrix, "S" for the rolling window and store
                levy_ll = Levy(price_panel=window_df)
                s_matrix = levy_ll.generate_levy_matrix()
                self.s[window_index] = s_matrix

                # Generate adjacency matrix, "A", convert to directed network, "G" and store
                a_matrix = np.maximum(s_matrix, 0)
                directed_net = nx.from_pandas_adjacency(a_matrix, create_using=nx.DiGraph)
                self.g[window_index] = directed_net
                
    
    def _calculate_return_panel(self):
        """
        Calculate the return panel based on the available scoring matrix "S".
        If the scoring matrix is not available, generate scores before calculating returns.
        """
        # Check if the scoring matrix is available
        if not self.s:
            # If not available, generate 
            self.generate_matrices_and_networks()

        # Calculate returns
        return_panel = self.price_panel.pct_change()
        
        # Find the earliest index from the scoring matrix "S" keys and filter the return panel
        earliest_idx = min(self.s.keys())
        self.return_panel = return_panel[earliest_idx:]
            
    
    def calculate_global_scores(self):
        """
        Calculate global scores for each asset based on the available scoring matrix "S".

        For every index in the S matrix, average every asset's lead-lag score with respect to all other assets
        as that asset's global score for that index. The scores are used for sorting assets from most likely 
        to be a leader to most likely to be a follower in constructing the global portfolio.
        """
        # Check if the scoring matrix is available
        if not self.s:
            # If not available, generate 
            self.generate_matrices_and_networks()
            
        global_scores = pd.DataFrame()
        for idx, s_matrix in self.s.items():
            dt_scores = pd.DataFrame({idx: s_matrix.mean(axis=1)}).T
            global_scores = pd.concat([global_scores, dt_scores])
            
        self.global_scores = global_scores
        
    
    def find_gp_leaders_followers(self, selection_percentile: float = 0.2):
        """
        Identify leaders and followers based on global ranking scores and the specified percentile.

        Parameters:
        - selection_percentile (float): The percentile used to determine leaders and followers.
        """
        # Check if the global scores are available
        if self.global_scores.empty:
            # If not, generate
            self.calculate_global_scores()
        
        # Utility function to find global leaders and followers for each index
        def identify_leaders_followers(row):
            non_nan_values = row.dropna()
            num_values = len(non_nan_values)
            follower_assets = non_nan_values.nsmallest(int(selection_percentile * num_values)).index
            leader_assets = non_nan_values.nlargest(int(selection_percentile * num_values)).index
            return follower_assets, leader_assets
        
        self.gp_leaders_followers = self.global_scores.apply(identify_leaders_followers, axis=1)
        
        
    def cluster_directed_nets(self, 
                              k_min: int = 3,
                              k_max: int = 10,
                              kmeans_init: str = 'k-means++',
                              kmeans_n_init: int = 10,
                              kmeans_random_state: int = 42,
                              show_progress: bool = True):
        """
        Clusters directed networks using the Hermitian clustering approach and stores the results.

        Parameters:
        - k_min (int): The minimum number of clusters for each network.
        - k_max (int): The maximum number of clusters for each network.
        - kmeans_init (str): Initialization method for KMeans. Default is 'k-means++'.
        - kmeans_n_init (int): Number of times KMeans will be run with different centroid seeds. Default is 10.
        - kmeans_random_state (int): Random state for KMeans. Default is 42.
        - show_progress (bool): Flag to show the progress bar.

        Note: Make sure to have the 'tqdm' library installed for the progress bar to work.
        """
        # Check if the directed networks are available
        if not self.g:
            # If not available, generate 
            self.generate_matrices_and_networks()
            
        # A dict of dicts, datetime and cluster number are keys, values are assets in each cluster    
        dt_cluster_dict = {}
        for dt, dt_g in tqdm(self.g.items(), 
                             desc='Clustering Lead-Lag Networks Using Hermitian Algorithm', 
                             disable=not show_progress):
            # Do Clustering
            clusterer = Hermitian(directed_net=dt_g)    
            dt_cluster_dict[dt] = clusterer.cluster_hermitian_opt(k_min=k_min,
                                                                  k_max=k_max, 
                                                                  kmeans_init=kmeans_init,
                                                                  kmeans_n_init=kmeans_n_init,
                                                                  kmeans_random_state=kmeans_random_state)
            
        self.dt_cluster_dict = dt_cluster_dict
        
    
    def find_cp_leaders_followers(self, selection_percentile: float = 0.2):
        """
        Identify leaders and followers based on the clustered lead-lag networks and the specified percentile.

        Parameters:
        - selection_percentile (float): The percentile used to determine leaders and followers.
        """        
        # Check availability of the backtest data
        if not self.dt_cluster_dict:
            raise ValueError("Clustered networks data is empty. Run 'cluster_directed_nets()' method with your desired parameters and try again.")
        
        dt_cl_lfs_dict = {}
        for dt, cluster_dict in self.dt_cluster_dict.items():
            
            # Slice score matrix for a given datetime/index
            dt_s = self.s[dt]

            # Do inter-cluster ranking and identify leaders and followers for each cluster for a given 
            cl_lfs_dict = {}
            for c_no, assets in cluster_dict.items():
                
                # Slice cluster assets from the dt_s matrix and calculate mean score of each asset
                dt_cluster_s = dt_s.loc[assets, assets]
                dt_cluster_score = dt_cluster_s.mean(axis=1)
                
                # Add clusters leaders/followers to a dict
                leaders = dt_cluster_score.nlargest(int(selection_percentile*len(dt_cluster_score))).index.tolist()
                followers = dt_cluster_score.nsmallest(int(selection_percentile*len(dt_cluster_score))).index.tolist()
                cl_lfs_dict[c_no] = {'leaders': leaders, 'followers': followers}
                
            # Add datetime'index clusters leader followr data
            dt_cl_lfs_dict[dt] = cl_lfs_dict
                
        self.cp_leaders_followers = dt_cl_lfs_dict
            
        
    
    def backtest_gp(self, selection_percentile: float = 0.2, show_progress: bool = True):
        """
        Performs the walk-forward backtest for the global portfolio by considering leaders and followers 
        identified based on global ranking scores. It utilizes the return panel, global leaders/followers, 
        and scoring matrices generated within the LeadLagPortfolio instance. The backtest results include 
        entry and exit dates, returns for leaders at time T, followers and market at time T+1, at each time 
        step, as well as cumulative portfolio returns for followers, the market, and the global portfolio.

        Parameters:
        - selection_percentile (float): The percentile used to identify leaders and followers in the global portfolio. 
                                        Defaults to 0.2.
        - show_progress (bool): If True, display a progress bar during the backtesting process. Defaults to True.

        Returns:
        None: The backtest results are stored in the 'gp_data' attribute of the LeadLagPortfolio object.
        """
        # Check if the global leadrs/followers are available or not
        if self.gp_leaders_followers.empty:
            # If not, find them
            self.find_gp_leaders_followers(selection_percentile = selection_percentile)
        
        # Check if the return panel is available or not
        if self.return_panel.empty:
            # If not, calculate it
            self._calculate_return_panel()
        
        # Define empty lists for gathering global portfolio backtest data 
        leaders_ret = []
        followers_ret = []
        mkt_ret = []
        entry_dt = []
        exit_dt = []
        
        # Find leaders, followers and market return, entry and exit datetimes for global portfolio
        for i in tqdm(range(len(self.gp_leaders_followers)),
                      desc='Generating Backtest Results for Global Portfolio', 
                      disable=not show_progress):
            
            # Entry and Exit datetime for positions
            entry_dt.append(self.return_panel.index[i])
            exit_dt.append(self.return_panel.index[i+1])
            
            # Global leaders and followers at time T 
            followers = self.gp_leaders_followers[i][0].tolist()
            leaders = self.gp_leaders_followers[i][1].tolist()
            
            # Leaders return at time T, will be used for buy/sell signal of the followers at time T+1
            leaders_ret.append(self.return_panel.iloc[i][leaders].mean())
    
            # Followers and market return at time T+1, will be used to calculate GP performance
            followers_ret.append(self.return_panel.iloc[i+1][followers].mean())
            mkt_ret.append(self.return_panel.iloc[i+1].dropna().mean())
        
        # Global portfolio data
        gp_data = pd.DataFrame({'Entry':entry_dt, 
                                'Exit':exit_dt, 
                                'LRet':leaders_ret, 
                                'FRet':followers_ret, 
                                'MktRet':mkt_ret})
        
        # Calculate global portfolio return
        gp_data['PRet'] = gp_data.apply(lambda x: x['FRet']-x['MktRet'] if x['LRet'] > 0 else x['MktRet']-x['FRet'], axis=1)
        
        # Calculate followers, market and global portfolio return
        gp_data['FPnL'] = (gp_data['FRet']+1).cumprod()
        gp_data['MktPnL'] = (gp_data['MktRet']+1).cumprod()
        gp_data['PnL'] = (gp_data['PRet']+1).cumprod()
        
        # Add "Date" index to dataframe
        gp_data['Date'] = gp_data['Exit']
        gp_data = gp_data.set_index('Date')
        
        self.gp_data = gp_data
        
        
    def _calc_clusters_weighted_return(self, t, t_plus_one):
        """
        Calculates all clusters portfolios as a single weighted portfolio
        """
        # Return and weight of each cluster's portfolio on the requested datetime
        cluster_returns = []
        cluster_weights = []
        
        # Iterate over the clusters, leaders/followers list of the given datetime/index
        for c_no, lf_dict in self.cp_leaders_followers[t].items():

            # Extract leaders/followers from lf_dict of the cluster
            cluster_leaders = lf_dict['leaders']
            cluster_followers = lf_dict['followers']
            
            # If there is at least one leader/follower, then continue
            if len(cluster_followers) > 0:
                cluster_universe = self.dt_cluster_dict[t][c_no]
                
                # Calculate leaders return at time T, followers and cluster universe return at time T+1
                cluster_leaders_ret = self.return_panel.loc[t][cluster_leaders].mean()
                cluster_followers_ret = self.return_panel.loc[t_plus_one][cluster_followers].mean()
                cluster_universe_ret = self.return_panel.loc[t_plus_one][cluster_universe].mean()
                
                # Clustered portfolio return
                if cluster_leaders_ret > 0:
                    cluster_return = cluster_followers_ret - cluster_universe_ret
                else:
                    cluster_return = cluster_universe_ret - cluster_followers_ret
                    
                # Cosider number of followers as that clusters's weight
                cluster_weight = len(cluster_followers) 
                cluster_weights.append(cluster_weight) 
                cluster_returns.append(cluster_return)
                    
        cp_return = np.dot(cluster_returns, cluster_weights) / sum(cluster_weights)
        return cp_return 
        
        
    def backtest_cp(self, selection_percentile: float = 0.2, show_progress: bool = True):
        """
        Performs the walk-forward backtest for the global portfolio by considering leaders and followers 
        identified based on global ranking scores. It utilizes the return panel, global leaders/followers, 
        and scoring matrices generated within the LeadLagPortfolio instance. The backtest results include 
        entry and exit dates, returns for leaders at time T, followers and market at time T+1, at each time 
        step, as well as cumulative portfolio returns for followers, the market, and the global portfolio.

        Parameters:
        - selection_percentile (float): The percentile used to identify leaders and followers in the global portfolio. 
                                        Defaults to 0.2.
        - show_progress (bool): If True, display a progress bar during the backtesting process. Defaults to True.

        Returns:
        None: The backtest results are stored in the 'gp_data' attribute of the LeadLagPortfolio object.
        """
        # Check if the global leadrs/followers are available or not
        if not self.cp_leaders_followers:
            # If not, find 
            self.find_cp_leaders_followers(selection_percentile = selection_percentile)
        
        # Check if the return panel is available or not
        if self.return_panel.empty:
            # If not, calculate it
            self._calculate_return_panel()
            
        # Define empty lists for gathering global portfolio backtest data 
        cp_ret = []
        entry_dt = []
        exit_dt = []
        
        # Find leaders, followers and market return, entry and exit datetimes for global portfolio
        dt_list = list(self.cp_leaders_followers.keys())
        
        for i in tqdm(range(len(dt_list)-1),
                      desc='Generating Backtest Results for Clustered Portfolio', 
                      disable=not show_progress):
            
            # Entry and Exit datetime for positions
            entry_dt.append(self.return_panel.index[i])
            exit_dt.append(self.return_panel.index[i+1])
            cp_ret.append(self._calc_clusters_weighted_return(t=dt_list[i],
                                                              t_plus_one=dt_list[i+1]))
            
        # Global portfolio data
        cp_data = pd.DataFrame({'Entry':entry_dt, 
                                'Exit':exit_dt, 
                                'PRet':cp_ret})
            
        cp_data['PnL'] = (cp_data['PRet']+1).cumprod()
        
        # Add "Date" index to dataframe
        cp_data['Date'] = cp_data['Exit']
        cp_data = cp_data.set_index('Date')
        
        self.cp_data = cp_data
        
    def backtest_gcp(self, selection_percentile: float = 0.2, show_progress: bool = True):
        """
        Performs the walk-forward backtest for the global clustered portfolio by considering leaders and followers 
        identified based on global ranking scores. It utilizes the return panel, global leaders/followers, 
        and scoring matrices generated within the LeadLagPortfolio instance. The backtest results include 
        entry and exit dates, returns for leaders at time T, followers and market at time T+1, at each time 
        step, as well as cumulative portfolio returns for followers, the market, and the global portfolio.

        Parameters:
        - selection_percentile (float): The percentile used to identify leaders and followers in the global portfolio. 
                                        Defaults to 0.2.
        - show_progress (bool): If True, display a progress bar during the backtesting process. Defaults to True.

        Returns:
        None: The backtest results are stored in the 'gp_data' attribute of the LeadLagPortfolio object.
        """
        # Check if the global leadrs/followers are available or not
        if not self.cp_leaders_followers:
            print('find_cp_leaders_followers is empty')
            # If not, find 
            self.find_cp_leaders_followers(selection_percentile = selection_percentile)
        
        # Check if the return panel is available or not
        if self.return_panel.empty:
            # If not, calculate it
            self._calculate_return_panel()
            
        # Define empty lists for gathering global portfolio backtest data 
        leaders_ret = []
        followers_ret = []
        mkt_ret = []
        entry_dt = []
        exit_dt = []
        
        # Find leaders, followers and market return, entry and exit datetimes for global portfolio
        dt_list = list(self.cp_leaders_followers.keys())
        
        # Find leaders, followers and market return, entry and exit datetimes for global portfolio
        for i in tqdm(range(len(self.cp_leaders_followers)),
                        desc='Generating Backtest Results for Global Clustered Portfolio', 
                        disable=not show_progress):
            # Entry and Exit datetime for positions
            entry_dt.append(self.return_panel.index[i])
            exit_dt.append(self.return_panel.index[i+1])
            
            # Aggregate leaders and followers from all clusters for a given day
            leaders = [leader for leaders_list in self.cp_leaders_followers[dt_list[i]].values() for leader in leaders_list['leaders']]
            followers = [follower for follower_list in self.cp_leaders_followers[dt_list[i]].values() for follower in follower_list['followers']]
            
            # Leaders return at time T, will be used for buy/sell signal of the followers at time T+1
            leaders_ret.append(self.return_panel.iloc[i][leaders].mean())

            # Followers and market return at time T+1, will be used to calculate GP performance
            followers_ret.append(self.return_panel.iloc[i+1][followers].mean())
            mkt_ret.append(self.return_panel.iloc[i+1].dropna().mean())
            
        # Global clustered portfolio data
        gcp_data = pd.DataFrame({'Entry':entry_dt, 
                                'Exit':exit_dt, 
                                'LRet':leaders_ret, 
                                'FRet':followers_ret, 
                                'MktRet':mkt_ret})

        # Calculate global portfolio return
        gcp_data['PRet'] = gcp_data.apply(lambda x: x['FRet']-x['MktRet'] if x['LRet'] > 0 else x['MktRet']-x['FRet'], axis=1)

        # Calculate followers, market and global portfolio return
        gcp_data['FPnL'] = (gcp_data['FRet']+1).cumprod()
        gcp_data['MktPnL'] = (gcp_data['MktRet']+1).cumprod()
        gcp_data['PnL'] = (gcp_data['PRet']+1).cumprod()

        # Add "Date" index to dataframe
        gcp_data['Date'] = gcp_data['Exit']
        gcp_data = gcp_data.set_index('Date')
        
        self.gcp_data = gcp_data
            
            
        
    def plot_portfolio_performance(self, rf: float = 0.02, start_dt: str = '2019-09-01', 
                                   end_dt: str = '2023-10-30', gcp: bool = True, 
                                   cp: bool = False, fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plots portfolio performance over time. The GP portfolio always present in the plot, but GCP and CP must be 
        set to True to be included.

        Parameters:
        - rf (float): Risk-free rate. Defaults to 0.02.
        - start_dt (str): Start date in the format 'YYYY-MM-DD'. Defaults to '2019-09-01'.
        - end_dt (str): End date in the format 'YYYY-MM-DD'. Defaults to '2023-10-30'.
        - gcp (bool): Include Global Clustered portfolio. Defaults to True.
        - cp (bool): Include Clustered portfolio. Defaults to False.
        - fig_size (Tuple[int, int], optional): Figure size as a tuple of width and height. Defaults to (10, 6).

        Returns:
        - fig (matplotlib.figure.Figure): Matplotlib figure containing the portfolio performance plot.
        """
        # Check availability of the backtest data
        if self.gp_data.empty:
            raise ValueError("Global portfolio backtest data is empty. Run 'backtest_gp()' method and try again.")
        
        if cp and self.cp_data.empty:
            raise ValueError("Clustered portfolio backtest data is empty. Run 'backtest_cp()' method and try again.")
        
        if gcp and self.gcp_data.empty:
            raise ValueError("Global Clustered portfolio backtest data is empty. Run 'backtest_gcp()' method and try again.")


        # Copy global portfolio backtest data
        gp_data = self.gp_data.copy()

        # Filter out requested dates and calculate PnL and DD series
        gp_data = gp_data[(gp_data['Entry'] >= start_dt) & (gp_data['Entry'] <= end_dt)]
        gp_data['PnL'] = (gp_data['PRet'] + 1).cumprod()
        gp_data['DD'] = (gp_data['PnL'] - gp_data['PnL'].cummax()) / gp_data['PnL'].cummax()
        
        # Annual volatility, return, Sharpe ratio, and maximum drawdown
        gp_ann_vol = np.std(gp_data['PRet']) * np.sqrt(365.25)
        gp_ann_ret = (gp_data['PnL'].iloc[-1]) ** (365.25 / len(gp_data)) - 1
        gp_ann_sr = (gp_ann_ret - rf) / gp_ann_vol
        gp_max_dd = gp_data['DD'].min()
        gp_metrics = ['GP', f'{gp_ann_ret*100:.1f}%', f'{gp_ann_vol*100:.1f}%',f'{gp_max_dd*100:.1f}%',f'{gp_ann_sr:.2f}']
        metrics = [gp_metrics]
            
        if cp:
            # Copy global portfolio backtest data
            cp_data = self.cp_data.copy()

            # Filter out requested dates and calculate PnL and DD series
            cp_data = cp_data[(cp_data['Entry'] >= start_dt) & (cp_data['Entry'] <= end_dt)]
            cp_data['PnL'] = (cp_data['PRet'] + 1).cumprod()
            cp_data['DD'] = (cp_data['PnL'] - cp_data['PnL'].cummax()) / cp_data['PnL'].cummax()
            
            # Annual volatility, return, Sharpe ratio, and maximum drawdown
            cp_ann_vol = np.std(cp_data['PRet']) * np.sqrt(365.25)
            cp_ann_ret = (cp_data['PnL'].iloc[-1]) ** (365.25 / len(cp_data)) - 1
            cp_ann_sr = (cp_ann_ret - rf) / cp_ann_vol
            cp_max_dd = cp_data['DD'].min()
            cp_metrics = ['CP', f'{cp_ann_ret*100:.1f}%', f'{cp_ann_vol*100:.1f}%',f'{cp_max_dd*100:.1f}%',f'{cp_ann_sr:.2f}']
            metrics.append(cp_metrics)
            
        if gcp:
            # Copy global portfolio backtest data
            gcp_data = self.gcp_data.copy()

            # Filter out requested dates and calculate PnL and DD series
            gcp_data = gcp_data[(gcp_data['Entry'] >= start_dt) & (gcp_data['Entry'] <= end_dt)]
            gcp_data['PnL'] = (gcp_data['PRet'] + 1).cumprod()
            gcp_data['DD'] = (gcp_data['PnL'] - gcp_data['PnL'].cummax()) / gcp_data['PnL'].cummax()
            
            # Annual volatility, return, Sharpe ratio, and maximum drawdown
            gcp_ann_vol = np.std(gcp_data['PRet']) * np.sqrt(365.25)
            gcp_ann_ret = (gcp_data['PnL'].iloc[-1]) ** (365.25 / len(gcp_data)) - 1
            gcp_ann_sr = (gcp_ann_ret - rf) / gcp_ann_vol
            gcp_max_dd = gcp_data['DD'].min()
            gcp_metrics = ['GCP', f'{gcp_ann_ret*100:.1f}%', f'{gcp_ann_vol*100:.1f}%',f'{gcp_max_dd*100:.1f}%',f'{gcp_ann_sr:.2f}']
            metrics.append(gcp_metrics)
            
        # Plot PnL and drawdown curves and add performance statistics
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[1])
        ax0 = plt.subplot(gs[0], sharex=ax1)

        # Subplot for PnL
        ax0.plot(gp_data.index, gp_data['PnL'], linewidth=1, label='GP')
        if cp:
            ax0.plot(cp_data.index, cp_data['PnL'], linewidth=1, label='CP')
        if gcp:
            ax0.plot(gcp_data.index, gcp_data['PnL'], linewidth=1, label='GCP')
            
        # Add metrics table
        table = ax0.table(cellText=metrics, loc='lower right', 
                          colLabels=['Portfolio', r'$Ret_{ann}$', r'$Vol_{ann}$', r'$DD_{max}$', r'$SR_{ann}$'], 
                          cellLoc='center', colColours=['#f3f3f3']*5)
        table.auto_set_font_size(True)
        table.scale(0.4, 1.4)
        
        # Add y-axis label, disable ticks and legend position 
        ax0.set_ylabel('Portfolio PnL')
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.legend(loc='upper left')
        
        # Subplot for Drawdown
        ax1.plot(gp_data.index, gp_data['DD']*100, linewidth=1)
        if cp:
            ax1.plot(cp_data.index, cp_data['DD']*100, linewidth=1)
        if gcp:
            ax1.plot(gcp_data.index, gcp_data['DD']*100, linewidth=1)
        ax1.set_ylabel('DD (%)')
        
        # Common title and adjust layout
        plt.suptitle('Portfolio Performance Over Time', fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    
    def plot_monthly_returns(self, portfolio: str = 'GP', start_dt: str = '2019-09-01', 
                             end_dt: str = '2023-10-30', fig_size: Tuple[int, int] = (10, 4)) -> plt.Figure:
        """
        Plots a heatmap of monthly and year-to-date (YTD) returns for a given portfolio.

        Parameters:
        - portfolio (str): Type of portfolio ('GP', 'CP', or 'GCP').
        - start_dt (str): Start date in the format 'YYYY-MM-DD'.
        - end_dt (str): End date in the format 'YYYY-MM-DD'.

        Returns:
        - fig: Matplotlib figure containing the heatmap.
        """
        # Check the requested portfolio type
        if portfolio == 'GP':
            portfolio_data = self.gp_data.copy()
            heatmap_title = 'Global'
        elif portfolio == 'CP':
            portfolio_data = self.cp_data.copy()
            heatmap_title = 'Clustered'
        elif portfolio == 'GCP':
            portfolio_data = self.gcp_data.copy()
            heatmap_title = 'Global Clustered'
        else:
            # Raise an error for unknown portfolio types
            raise ValueError("Invalid entry for portfolio type. Portfolio type must be 'GP', 'GCP', or 'CP'.")
        
        # Filter out according to start and end datetime
        portfolio_data = portfolio_data[(portfolio_data['Entry'] >= start_dt) & (portfolio_data['Entry'] <= end_dt)]

        # Calculate monthly returns
        monthly_ret = pd.DataFrame(portfolio_data['PRet'].resample('M').agg(lambda x: (1 + x).prod() - 1))
        monthly_ret['Year'] = monthly_ret.index.year
        monthly_ret['Month'] = monthly_ret.index.strftime('%b')
        heatmap_data = monthly_ret.pivot_table(values='PRet', index='Year', columns='Month', margins_name='All')

        # Calculate YTD using monthly data
        heatmap_data.fillna(0, inplace=True)
        heatmap_data_percentage = heatmap_data + 1
        heatmap_data['YTD'] = (heatmap_data_percentage.cumprod(axis=1).iloc[:, -1] - 1)
        heatmap_data.replace(0, np.nan, inplace=True)

        # Multiply in 100 to get percent returns
        heatmap_data = 100 * heatmap_data
        
        # Order the dataframe months and YTD
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','YTD']
        heatmap_data = heatmap_data.reindex(month_order, axis=1)

        # Plot portfolio monthly returns heatmap with YTD
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [12, 1]}, figsize=fig_size, sharey=True)

        # Plot monthly and YTD heatmaps
        sns.heatmap(heatmap_data.drop(columns=['YTD']), annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5, center=0, ax=ax1)
        sns.heatmap(heatmap_data[['YTD']], annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5, center=0, ax=ax2)

        # Remove x/y lables for YTD
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        # Plot title and other setings
        plt.tight_layout()
        plt.suptitle(f'{heatmap_title} Portfolio Monthly and YTD Returns', fontsize=11, fontweight='bold', y=1.05);
        
        return fig
       