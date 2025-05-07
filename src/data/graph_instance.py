import torch
import torch_geometric
import pandas as pd
import numpy as np
import h5py
import obspy.core as oc
from geopy.distance import geodesic, distance
import networkx as nx
import matplotlib.pyplot as plt

import warnings
import random
import requests

warnings.filterwarnings('ignore')


class GIDataset(torch.utils.data.Dataset):
    """
    Class for reading the Graph Instance Dataset.

    Parameters
    ----------
    meta_csv_path : str or pathlib.Path object
        path to the csv file containing the metadata.
    distances_npy_path : str or pathlib.Path object
        path to the npy file containing the distances between each pair of stations.
    h5_path : str or pathlib.Path object
        path to the hdf5 file containing the waveform data.
    split_path : str or pathlib.Path object
        path to the txt file containing the train, validation or test split.
    ts_length : int (Default: 10)
        length of the desired input time series in seconds.
    edge_cutoff : float (Default: 0.0)
        threshold for cutting edges in the graph according to Bloemheuvel et al. 2023. Has to be in [0,1].
    norm : bool (Default: True)
        whether to norm the time series of a sample by the maximum value
    dropout : float (Default: 0.0)
        fraction of stations to mask out (except for the first one to receive the signal).
    graph_format : str (Default: 'complete')
        which type of graph to return for a sample. 'complete' does not filter out edges.
        'reduced' removes low-weighted edges as long as the graph stays a single component.
    min_node_degree : int (Default: 1)
        minimum node degree in the graph if low-weighted edges are removed (i.e. graph='reduced).
    k_hop_reachability: int (Default: None)
        maximum diameter that the graph is allowed to have. E.g. 2 will guarantee that each node can reach each other
        node in the graph within 2 steps of message passing.
    pad_ts_s : float / int (Default: 0.0)
        If set to 0, the time series time window starts from the first P arrival time. Otherwise, it starts a
        random number of seconds [0, pad_ts_s] before the first P arrival time.
    deterministic : bool (Default: False)
        if set to True, the source_id is set as a random seed before drawing a random number in [0, pad_ts_s] to shift
        the time series. This means each sample is always returned the same way.
    verbose : bool (Default: False)
        If set to true, prints out information about the dataset such as data on the graph and plots the graph.

    Attributes
    ----------
    h5_path : str or pathlib.Path object
        path to the hdf5 file containing the waveform data.
    split_path : str or pathlib.Path object
        path to the txt file containing the train, validation or test split.
    meta_data_df : pandas.DataFrame
        contains the metadata of the dataset according to the documentation of the Instance dataset.
    source_id_list : list of str
        contains the IDs of all earthquake events in the dataset according to the Istituto Nazionale di Geofisica e
        Vulcanologia (INGV).
    station_id_list : list of str
        contains the IDs of all stations according to the International Registry of Seismograph Stations, IR.
    distances : numpy.ndarray
        contains the distances in km between each pair of stations.
    edge_cutoff : float (Default: 0.0)
        threshold for cutting edges in the graph according to Bloemheuvel et al. 2023. Has to be in [0,1].
    ts_length : int (Default: 10)
        length of the desired input time series in seconds.
    frequency : int or float (Default: 100)
        sampling frequency in Hz.
    n_channels : int (Default: 3)
        number of channels per recording (E:east-west, N:north-south, Z:up-down).
    pad : bool (Default: True)
        whether to pad the time series to the desired length if needed.
    fill_value : float (Default: 0)
        value to fill missing values with during padding.
    differentiate_velocities : bool (Default: True)
        whether to differentiate velocities to accelerations for HH and EH channels.
    norm : bool (Default: True)
        whether to norm the time series of a sample by the maximum value
    dropout : float (Default: 0.0)
        fraction of stations to mask out (except for the first one to receive the signal).
    graph_format : str (Default: 'complete')
        which type of graph to return for a sample. 'complete' does not filter out edges.
        'reduced' removes low-weighted edges as long as the graph stays a single component.
    min_node_degree : int (Default: None)
        minimum node degree in the graph if low-weighted edges are removed (i.e. graph='reduced).
    k_hop_reachability: int (Default: None)
        maximum diameter that the graph is allowed to have. E.g. 2 will guarantee that each node can reach each other
        node in the graph within 2 steps of message passing.
    pad_ts_s : float / int (Default: 0.0)
        If set to 0, the time series time window starts from the first P arrival time. Otherwise, it starts a
        random number of seconds [0, pad_ts_s] before the first P arrival time.
    deterministic : bool (Default: False)
        if set to True, the source_id is set as a random seed before drawing a random number in [0, pad_ts_s] to shift
        the time series. This means each sample is always returned the same way.
    random_ts_padding : bool (Default: True)
        if set to True, the time series time window starts randomly 0-pad_ts_s before the first P arrival time. If set
        to False, the time window starts exactly pad_ts_s before the first P arrival time. Masking of stations for
        station dropout will remain random if the deterministic option is set to False.
    verbose : bool (Default: False)
        If set to true, prints out information about the dataset such as data on the graph and plots the graph.
    """

    def __init__(self, meta_csv_path, distances_npy_path, h5_path, split_path=None, ts_length=10, edge_cutoff=0.0,
                 norm=True, dropout=0.0, graph_format='complete', min_node_degree=None, k_hop_reachability=None,
                 pad_ts_s=0, deterministic=False, random_ts_padding=True, verbose=False):
        self.h5_path = h5_path
        self.split_path = split_path
        self.meta_data_df = self.get_meta_data(meta_csv_path)
        self.source_id_list = sorted(self.meta_data_df['source_id'].unique())
        self.filter_for_split()  # if a split is given, filter out all other events not contained in the split
        self.station_id_list = sorted(self.meta_data_df['station_code'].unique())
        self.distances = np.load(distances_npy_path)
        self.edge_cutoff = edge_cutoff  # cut edges with lower weight in the graph
        self.ts_length = ts_length  # length of time series in seconds
        self.frequency = 100  # sampling frequency in Hz
        self.n_channels = 3  # number of channels per recording: 3 (E:east-west, N:north-south, Z:up-down)
        self.pad = True  # pad time series which are not complete within
        self.fill_value = 0  # value to pad shorter time series with
        self.differentiate_velocities = True  # whether to differentiate velocities to accelerations
        self.norm = norm  # whether to norm the time series of a sample
        self.dropout = dropout  # ratio of stations to mask out
        self.graph_format = graph_format  # which type of graph to use (complete, reduced)
        self.min_node_degree = min_node_degree  # minimum node degree for the graph
        self.k_hop_reachability = k_hop_reachability  # maximum graph diameter
        self.pad_ts_s = pad_ts_s  # pad the time series start with up to pad_ts_s seconds
        self.deterministic = deterministic  # whether the time series sampling should be deterministic
        self.random_ts_padding = random_ts_padding  # whether the time series padding should be random
        self.verbose = verbose  # whether to print infos e.g. about the graph or plot the graph structure

    def _print(self, *args, **kwargs):
        """
        Print function for verbose output (e.g. for the graph construction)

        Parameters
        ----------
        x : object with __repr__ / __str__ method
            object to be printed.

        Returns
        -------
        None
        """
        if self.verbose:
            print(*args, **kwargs)

    def __len__(self):
        """
        Return the number of earthquake samples in the dataset.

        Parameters
        ----------

        Returns
        -------
        length : int
            number of samples in the dataset.
        """
        length = len(self.source_id_list)
        return length

    @staticmethod
    def get_meta_data(meta_csv_path):
        """
        Reads the dataset metadata from a csv file and preprocesses it.

        Parameters
        ----------
        meta_csv_path: str or pathlib.Path object
            path to the csv file containing the metadata.

        Returns
        -------
        meta_data_df : pandas.DataFrame
            contains the metadata of the dataset according to the documentation of the Instance dataset.
        """
        meta_data_df = pd.read_csv(meta_csv_path)
        # Convert argument of metadata columns with different type to numeric
        list_eve = ['trace_E_min_counts', 'trace_N_min_counts', 'trace_Z_min_counts',
                    'trace_E_max_counts', 'trace_N_max_counts', 'trace_Z_max_counts',
                    'trace_E_median_counts', 'trace_N_median_counts', 'trace_Z_median_counts',
                    'trace_E_mean_counts', 'trace_N_mean_counts', 'trace_Z_mean_counts',
                    'trace_E_pga_perc', 'trace_N_pga_perc', 'trace_Z_pga_perc',
                    'trace_E_pga_cmps2', 'trace_N_pga_cmps2', 'trace_Z_pga_cmps2',
                    'trace_E_pgv_cmps', 'trace_N_pgv_cmps', 'trace_Z_pgv_cmps',
                    'trace_E_snr_db', 'trace_N_snr_db', 'trace_Z_snr_db',
                    'trace_E_sa03_cmps2', 'trace_N_sa03_cmps2', 'trace_Z_sa03_cmps2',
                    'trace_pgv_cmps', 'trace_pga_perc',
                    'trace_EQT_number_detections', 'trace_EQT_P_number', 'trace_EQT_S_number', 'trace_GPD_P_number',
                    'trace_GPD_S_number']
        for ele in list_eve:
            meta_data_df[ele] = pd.to_numeric(meta_data_df[ele], errors='coerce')
        return meta_data_df

    def filter_for_split(self):
        """
        Filters the source id list (i.e. the earthquake ids) to only contain those belonging to the respective split, if
        one is provided. Otherwise, the function does nothing.

        Parameters
        ----------

        Returns
        -------

        """
        # Filter source id list for respective split (if given)
        if self.split_path:
            # read split txt file
            with open(self.split_path, 'r') as f:
                indices = f.readlines()
            indices = [int(line.strip()) for line in indices]
            # filter source id list
            self.source_id_list = [self.source_id_list[i] for i in indices]

    def build_stream(self, df_row, h5, differentiate_velocities=True):
        """
        Collects waveform data and its metadata for a single line from the Graph Instance dataset. A single line refers
        to the channel (EH/HN/HH) of a single station for a specific event. Returns a single stream.

        Parameters
        ----------
        df_row : pandas.Series
            single row from the metadata DataFrame.
        h5 : h5py.File object
            handle to the hdf5 file containing the waveform data.
        differentiate_velocities : bool (Default: True)
            whether to differentiate velocities to obtain accelerations for all EH/HH channels.

        Returns
        -------
        st : obspy.core.stream.Stream object
            Each stream corresponds to one station. Each trace in a stream correspond to one channel (East-West E,
            North-South N, Up-Down Z). The traces are trimmed to their overlapping timeframe.
        """
        # create a Stats object to hold additional header information of an ObsPy Trace object
        stats = oc.Stats()
        # number of datapoints --> 100 = 1s of waveform data
        stats.npts = 12000
        # sample distance in seconds, this also infers stats.sampling_rate as the inverse
        stats.delta = df_row[
            'trace_dt_s'].values[0]
        # date and time of the first data sample given in UTC
        # pandas converts the timestring to np.datetime64. The precision (e.g. nanoseconds) might not be compatible with
        # obspy therefore, convert it to milliseconds ('M8[ms]') and then to an object ('O'), which makes it compatible
        # with ObsPy.Core.UTCDateTime:
        stats.starttime = oc.UTCDateTime(
            pd.to_datetime(df_row['trace_start_time']).values[0].astype('M8[ms]').astype('O'))
        stats.network = df_row['station_network_code'].values[0]
        stats.station = df_row['station_code'].values[0]
        # read waveform data
        wav_name = df_row['trace_name'].values[0]
        waveform = h5['data'][wav_name]  # shaped 3 x 12000 (3 orientations, 12000 datapoints)
        # create stream object (list like object of multiple ObsPy Trace objects)
        st = oc.Stream()
        orientations = 'ENZ'
        for i in range(self.n_channels):  # E: East-West, N: North-South, Z: Up-Down --> 3 traces per channel
            # create trace object for this orientation, add waveform data and metadata (stats)
            tr = oc.Trace()
            tr.data = waveform[i]
            # differentiate velocities to get accelerations if desired
            if differentiate_velocities and df_row['station_channels'].values[0] in ('EH', 'HH'):
                tr.differentiate(method='gradient')
            tr.stats = stats.copy()
            # add E, N, or Z (orientation) to the channel name
            tr.stats.channel = df_row['station_channels'].values[0] + orientations[i]
            # append trace to the stream
            st.append(tr)
        return st

    def get_streams_ims_by_idx(self, idx, differentiate_velocities=True):
        """
        Collects the waveform data as streams, intensity measurements (IMs) and metadata for a single earthquake event.

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDataset)-1].
        differentiate_velocities : bool (Default: True)
            whether to differentiate velocities to obtain accelerations for all EH/HH channels.

        Returns
        -------
        streams : list of obspy.core.stream.Stream objects
            List of all stream objects for the event. Each stream corresponds to one station. Each trace in a stream
            correspond to one channel (East-West E, North-South N, Up-Down Z).
        ims : torch.tensor
            targets as a tensor of shape [N, 5], where N is the number of stations and 5 is the number of intensity
            measurements used as targets. These correspond to Peak Ground Acceleration PGA (cm/s^2), Peak Ground
            Velocity PGV (cm/s), and Peak Spectral Acceleration SA at 0.3s, 1.0s and 3.0s Periods (cm/s^2).
        source_df : pandas.DataFrame
            dataframe containing the metadata for the earthquake event.
        event_stations : list of str
            list of length N, where N is the number of stations in the graph, with the station names as strings
            (according to the International Registry of Seismograph Stations, IR), which recorded the event.
        channels : list of str
            list of length N, where N is the number of channels in the graph, with the channel names (as strings), which
            were used to derive the input time series x. The channel names identify the sampling and instrument gain
            (e.g. HN, HH, EH).
        """
        # get source ID
        source_id = self.source_id_list[idx]
        # get source dataframe
        source_df = self.meta_data_df[self.meta_data_df['source_id'] == source_id]
        # remove rows where IMs contain NaN
        source_df = source_df.dropna(subset=['trace_pga_cmps2',
                                             'trace_pgv_cmps',
                                             'trace_sa03_cmps2',
                                             'trace_sa10_cmps2',
                                             'trace_sa30_cmps2'])
        # get list of stations which measured the event
        event_stations = source_df['station_code'].unique()
        # create a list of streams, one for each station
        streams = []
        # create a list of channels. These are just the EH/HH/HN code to know the origin of the data
        channels = []
        # Lists for intensity measurements
        pga_cmps2 = []
        pgv_cmps = []
        sa03_cmps2 = []
        sa10_cmps2 = []
        sa30_cmps2 = []
        # List for df rows that are used (only one of EH/HH/HN is used)
        source_df_rows = []
        with h5py.File(self.h5_path, 'r') as h5:
            for i, station in enumerate(event_stations):
                station_df = source_df[source_df['station_code'] == station]
                if 'HN' in list(station_df['station_channels']):
                    channel = 'HN'
                elif 'HH' in list(station_df['station_channels']):
                    channel = 'HH'
                elif 'EH' in list(station_df['station_channels']):
                    channel = 'EH'
                else:
                    raise Exception(f'No EH/HN/HH channel found for station {station} in sample {idx}.')
                channels.append(channel)
                # get appropriate row from the metadata dataframe
                df_row = station_df[station_df['station_channels'] == channel]
                source_df_rows.append(df_row)
                # read data
                stream = self.build_stream(df_row, h5, differentiate_velocities=differentiate_velocities)
                streams.append(stream)
                # get intensity measurements
                pga_cmps2.append(df_row['trace_pga_cmps2'].values[0])
                pgv_cmps.append(df_row['trace_pgv_cmps'].values[0])
                sa03_cmps2.append(df_row['trace_sa03_cmps2'].values[0])
                sa10_cmps2.append(df_row['trace_sa10_cmps2'].values[0])
                sa30_cmps2.append(df_row['trace_sa30_cmps2'].values[0])
        # stack IMs into single array and convert it to a tensor
        ims = torch.from_numpy(
            np.stack([np.array(x) for x in [pga_cmps2, pgv_cmps, sa03_cmps2, sa10_cmps2, sa30_cmps2]], axis=0)).type(
            torch.float32).t()
        # stack source df rows to a new, filtered source_df
        source_df = pd.concat(source_df_rows, ignore_index=True)
        return streams, ims, source_df, event_stations, channels

    def trim_streams(self, streams, source_df):
        """
        Trim a list of streams to the desired time series length, starting at earthquake origin time.

        Parameters
        ----------
        streams : list of obspy.core.stream.Stream objects
            List of all stream objects for the event. Each stream corresponds to one station. Each trace in a stream
            correspond to one channel (East-West E, North-South N, Up-Down Z).
        source_df : pandas.DataFrame
            DataFrame containing the event metadata, including the origin time of the earthquake.

        Returns
        -------
        trimmed_streams : list of obspy.core.stream.Stream objects
            Contains the streams trimmed to the desired length of the time series, starting at the origin time of the
            earthquake. Shorter time series are padded if desired.
        """
        # determine earliest arrival time for the P(rimary) wave or use the source_origin_time
        p_arrival_times = [oc.UTCDateTime(x) for x in
                           source_df['trace_P_arrival_time']]  # use trace_P_arrival_time or source_origin_time
        starttime = min(p_arrival_times)

        # Masking out stations
        # randomly set the start of the timeseries before the first P arrival time
        if self.deterministic:
            seed = int(source_df['source_id'].iloc[0])
            random.seed(seed)
        # Find the index of the first station to receive the signal
        first_station_idx = p_arrival_times.index(min(p_arrival_times))
        # Get indices of the remaining stations
        remaining_indices = [i for i in range(len(streams)) if i != first_station_idx]
        # Randomly select 20% of the remaining stations
        num_to_mask = int(len(streams) * self.dropout)
        mask_indices = random.sample(remaining_indices, min(num_to_mask, len(remaining_indices)))
        # set selected streams to 0
        masked = np.zeros(len(streams))
        for idx in mask_indices:
            for trace in streams[idx]:
                trace.data[:] = 0.
                pass
            # array to keep track which streams were masked
            masked[idx] = 1.

        # randomly set the start of the timeseries before the first P arrival time
        if self.deterministic and self.random_ts_padding:
            seed = int(source_df['source_id'].iloc[0])
            random.seed(seed)
            pad_ts = random.uniform(0, self.pad_ts_s)
        elif not self.deterministic and self.random_ts_padding:
            pad_ts = random.uniform(0, self.pad_ts_s)
        elif not self.random_ts_padding:
            pad_ts = self.pad_ts_s
        else:
            raise ValueError('Invalid Argument for time series padding.')
        self._print(f'Padding of the time series: {pad_ts}s')
        starttime = starttime - pad_ts
        stoptime = starttime + self.ts_length
        # trim the time series to the [starttime, stoptime] interval
        trimmed_streams = []
        for stream in streams:
            trimmed_stream = stream.trim(starttime, stoptime, pad=self.pad, fill_value=self.fill_value)
            trimmed_streams.append(trimmed_stream)
        return trimmed_streams, masked

    def streams_to_tensor(self, streams):
        """
        Converts a list of streams to a PyTorch tensor

        Parameters
        ----------
        streams : list of obspy.core.stream.Stream objects
            List of all stream objects for the event. Each stream corresponds to one station. Each trace in a stream
            correspond to one channel (East-West E, North-South N, Up-Down Z). The traces have to be of the same length.

        Returns
        -------
        arr : torch.tensor
            input time series for an event in a tensor of shape [N, T, C], where N is the number of stations that
            recorded the earthquake, T is the length of the time series, C is the number of channels (E,N,Z).
        """
        arr = np.zeros((len(streams), self.ts_length * self.frequency, self.n_channels))
        for i1, stream in enumerate(streams):
            for i2, trace in enumerate(stream):
                ts = trace.data
                arr[i1, :, i2] = ts[:self.ts_length * self.frequency]
        arr = torch.from_numpy(arr).type(torch.float32)
        # normalize if desired
        if self.norm:
            arr = arr / torch.max(torch.abs(arr))
        return arr

    def construct_graph(self, source_df, event_stations, export_graph_path=None):
        """
        Construct the graph for a single event.

        Parameters
        ----------
        source_df : pandas.DataFrame
            dataframe containing the metadata for a single earthquake event.
        event_stations : list of str
            list of length N, where N is the number of stations in the graph, with the station names as strings
            (according to the International Registry of Seismograph Stations, IR), which recorded the event.
        export_graph_path : str or pathlib.Path or None
            if not None, the edges of the graph will be exported to the given path as a csv file containing the edge
            endpoints as WKT LINESTRING that can be interpreted in GIS to visualize the graph.

        Returns
        -------
        L : torch.tensor
            normalized graph laplacian for the sample.
        """
        # Adapted from https://github.com/StefanBloemheuvel/GCNTimeseriesRegression/blob/main/graph_maker.py
        graph = nx.Graph()

        # Add nodes (stations) to the graph
        for station_code in event_stations:
            row = source_df[source_df['station_code'] == station_code].iloc[0]
            graph.add_node(row['station_code'], pos=(row['station_longitude_deg'], row['station_latitude_deg']))

        # Add edges (with geodesic distance) to the graph
        for s1 in event_stations:
            for s2 in event_stations:
                # find the correct entry in the distance matrix
                if s1 == s2:  # filters out self-loops
                    continue
                i1 = self.station_id_list.index(s1)
                i2 = self.station_id_list.index(s2)
                distance = self.distances[i1, i2]
                graph.add_edge(s1, s2, weight=distance, added_info=distance)

        # Make a new graph with weights higher for stations that are close to each other
        edge_list = nx.to_pandas_edgelist(graph)
        # normalize the distances
        edge_list['weight'] = (edge_list['weight'] - min(edge_list['weight'])) / (
                max(edge_list['weight']) - min(edge_list['weight']))
        # Large distances -> small weights
        edge_list['weight'] = 1.0 - edge_list[
            'weight']  # Bloemheuvel used 0.98 (in GitHub) instead of 1.0 (as in their paper)
        # Make adjacency matrix
        adj = nx.from_pandas_edgelist(edge_list, edge_attr=['weight'])
        adj = pd.DataFrame(nx.adjacency_matrix(adj, weight='weight').todense())
        # print(adj)
        # apply threshold
        adj[adj < self.edge_cutoff] = 0
        # create new graph with updated edge weights
        newgraph = nx.from_pandas_adjacency(adj)
        # relabel the nodes
        name_dict = dict(zip([x for x in range(len(event_stations))], [x for x in graph.nodes()]))
        newgraph = nx.relabel_nodes(newgraph, name_dict)
        # add longitude/latitude to new graph nodes
        nx.set_node_attributes(newgraph, nx.get_node_attributes(graph, 'pos'), 'pos')
        # add distance in km to new graph edges
        nx.set_edge_attributes(newgraph, nx.get_edge_attributes(graph, 'added_info'), 'added_info')

        self._print(newgraph)

        # reduce graph if desired
        if self.graph_format == 'reduced':
            edges = list(newgraph.edges())
            # sort edges by their weights
            edges = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'])

            for edge in edges:
                # print(u, v)
                u, v, attributes = edge
                weight = attributes['weight']
                added_info = attributes['added_info']
                # weight = newgraph[u][v]['weight']
                # added_info = newgraph[u][v]['added_info']
                # print(u, v, weight, added_info)
                # print(u, v, weight, added_info)
                # remove the edge
                newgraph.remove_edge(u, v)
                # STOPPING CRITERIA
                # 1) check if graph is no longer connected
                if not nx.is_connected(newgraph):
                    # add the last edge back to keep the graph connected
                    newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                    self._print('Edge added back in (Splitting of Component): ', u, v, newgraph[u][v])
                    break
                # 2) check if each node still has the minimum node degree
                if self.min_node_degree is not None:
                    if newgraph.degree[u] < self.min_node_degree or newgraph.degree[v] < self.min_node_degree:
                        # add the last edge back to maintain the minimum edge degree
                        newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                        self._print('Edge added back in (Min Node Degree): ', u, v, newgraph[u][v])
                        break
                # 3) check if the graph still has a diameter below the desired limit
                if self.k_hop_reachability is not None:
                    if nx.diameter(newgraph) > self.k_hop_reachability:
                        # add the last edge back to keep the graph diameter
                        newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                        self._print('Edge added back in (k-hop reachability): ', u, v, newgraph[u][v])
                        break
                # Old Code without graph diameter
                # if not nx.is_connected(newgraph) or newgraph.degree[u] < self.min_node_degree or newgraph.degree[
                #     v] < self.min_node_degree:
                #     # add the last edge back to keep the graph connected
                #     newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                #     self._print('Edge added back in: ', u, v, newgraph[u][v])
                #     break
        elif self.graph_format == 'complete':
            pass
        else:
            raise ValueError('Graph format not recognized')
        self._print(newgraph)

        # Export the graph as a csv file if desired
        if export_graph_path:
            self.graph_to_wkt(newgraph, source_df, export_graph_path)

        # ----------------------------------------------------------------------------------------------------------------
        # print some graph info
        edges1 = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'], reverse=False)[0][0]
        edges2 = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'], reverse=False)[0][1]
        edges3 = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'], reverse=False)[0][2]
        self._print('---------------------------------------------------------------------------------')
        self._print(f'edge with lowest weight = {edges1, edges2, edges3}, OG km = {graph[edges1][edges2]}')
        self._print(
            f'Average degree of the graph =  {np.mean([val for (node, val) in sorted(newgraph.degree(), key=lambda pair: pair[0])])}')

        degree_centralities = []
        for i in nx.degree_centrality(newgraph).values():
            degree_centralities.append(i)
        self._print('avg degree centrality = ', np.mean(degree_centralities))

        distances_og = []
        for i in newgraph.edges(data=True):
            distances_og.append(i[2]['added_info'])
        self._print('average distance og = ', np.array(distances_og).mean())

        # plot graph
        if self.verbose:
            plot_weights = [newgraph[u][v]['weight'] for u, v in newgraph.edges()]
            nx.draw(newgraph, nx.get_node_attributes(newgraph, 'pos'), with_labels=True, width=plot_weights)
            plt.show()
        # ----------------------------------------------------------------------------------------------------------------

        # save results
        # Calculate normalized laplacian of the graph (done automatically in torch geometric)
        # D = np.diag(np.sum(adj, axis=1))  # degree matrix D
        # I = np.identity(adj.shape[0])  # identity matrix I
        # D_inv_sqrt = np.linalg.inv(np.sqrt(D))  # D^-0.5
        # L = torch.from_numpy(I - np.dot(D_inv_sqrt, adj).dot(D_inv_sqrt))  # normalized graph laplacian
        # Make adjacency matrix of the filtered graph
        new_edge_list = nx.to_pandas_edgelist(newgraph)
        new_adj = nx.from_pandas_edgelist(new_edge_list, edge_attr=['weight'])
        new_adj = pd.DataFrame(nx.adjacency_matrix(new_adj, weight='weight').todense())

        new_adj = new_adj.to_numpy()
        new_adj = torch.from_numpy(new_adj)
        return new_adj

    def graph_to_wkt(self, graph, source_df, path):
        """
        Save the edges of the graph in WKT format as a csv file, which can be interpreted in GIS.

        Parameters
        ----------
        graph : networkx.Graph
            graph to save as a csv file
        source_df : pandas.DataFrame
            dataframe containing the meta data (for lat/lon) of the sample
        path : pathlib.Path or str
            path were to store the csv file

        Returns
        -------
        None
        """
        # create list of rows for the csv file
        rows = ['station1;station2;weight;distance_km;gis_linestring\n']
        # get edges
        edges = graph.edges(data=True)
        for edge in edges:
            # get nodes and weights from the edge
            n1, n2, attr = edge
            # get lat/lon of both nodes
            row1 = source_df[source_df['station_code'] == n1]
            lon1 = row1['station_longitude_deg'].values[0]
            lat1 = row1['station_latitude_deg'].values[0]
            row2 = source_df[source_df['station_code'] == n2]
            lon2 = row2['station_longitude_deg'].values[0]
            lat2 = row2['station_latitude_deg'].values[0]
            # create a row in WKT format for this edge
            row = f'{n1};{n2};{attr["weight"]};{attr["added_info"]};LINESTRING({lon1} {lat1}, {lon2} {lat2})\n'
            # add as row for the csv file
            rows.append(row)
        # export as csv file
        with open(path, 'w') as f:
            f.writelines(rows)

    def get_node_elevations(self, source_df):
        """
        Get the normalized elevations of the nodes in the graph.

        Parameters
        ----------
        source_df : pandas.DataFrame
            dataframe with the sample metadata including the elevations.

        Returns
        -------
        elevations : torch.tensor
            elevations of the nodes in the graph.
        """
        # filter source_df for unique stations
        source_df = source_df.drop_duplicates(subset=['station_code'])
        # get elevations in m
        elevations = torch.tensor(source_df['station_elevation_m'].values, dtype=torch.float32)
        elevations /= elevations.max()
        return elevations

    def get_node_positions(self, source_df):
        """
        Get relative, normalized positions of the nodes in the graph and the earthquake epicenter relative to a random
        reference point.

        Parameters
        ----------
        source_df : pandas.DataFrame
            dataframe containing the metadata of the sample, including station and earthquake positions.

        Returns
        -------
        node_dists : torch.tensor
            contains the positions of each node in the graph relative to a randomly chosen reference point within
            min/max of station latitudes and longitudes. The relative positions in km in north/south and east/west
            direction are normalized by the maximum absolute distance component.
        epi_dists : torch.tensor
            same as node_dists, but for the normalized relative position of the earthquake epicenter.
        norm_factor : float
            normalization factor used to normalize the relative positions in km.
        ref_lat_lon : tuple(float, float)
            latitude and longitude of the randomly chosen reference point.
        """
        # filter source_df for unique stations
        source_df = source_df.drop_duplicates(subset=['station_code'])
        # get latitudes and longitudes of stations and epicenter
        lats = torch.tensor(source_df['station_latitude_deg'].values, dtype=torch.float32)
        lons = torch.tensor(source_df['station_longitude_deg'].values, dtype=torch.float32)
        epi_lat = torch.tensor(source_df['source_latitude_deg'].values[0], dtype=torch.float32)
        epi_lon = torch.tensor(source_df['source_longitude_deg'].values[0], dtype=torch.float32)
        self._print(f'Min/Max Node Latitude: {lats.min()}/{lats.max()}; Longitude: {lons.min()}/{lons.max()}')
        # get random reference point
        ref_lat = random.uniform(min(lats), max(lats))
        ref_lon = random.uniform(min(lons), max(lons))
        ref_lat_lon = torch.tensor((ref_lat, ref_lon), dtype=torch.float32)

        # get distances in x (lon) and y (lat) direction between nodes and reference point
        def get_dists(ref_lat, ref_lon, lat, lon):
            # calculate distance along the latitude (North-South)
            dist_north_south_km = geodesic((ref_lat, ref_lon), (lat, ref_lon)).km
            # calculate distance along the longitudes (East-West)
            dist_east_west_km = geodesic((ref_lat, ref_lon), (ref_lat, lon)).km
            # determine the sign based on the relative position
            if lat < ref_lat:  # if node is further south (smaller lat) than reference point
                dist_north_south_km *= -1
            if lon < ref_lon:  # if node is further west (smaller lon) than reference point
                dist_east_west_km *= -1
            return dist_north_south_km, dist_east_west_km

        node_dists = torch.tensor([list(get_dists(ref_lat, ref_lon, lat, lon)) for lat, lon in zip(lats, lons)],
                                  dtype=torch.float32)
        # get distance between epicenter and reference point
        epi_dists = torch.tensor(get_dists(ref_lat, ref_lon, epi_lat, epi_lon), dtype=torch.float32)
        # get normalization factor as max distance component in km
        norm_factor = torch.tensor(node_dists.abs().max(), dtype=torch.float32)
        # normalize distances
        node_dists = node_dists / norm_factor
        epi_dists = epi_dists / norm_factor
        return node_dists, epi_dists, norm_factor, ref_lat_lon

    @staticmethod
    def get_absolute_position(position_ns_ew, ref_lat_lon, norm_factor):
        """
        Calculates the latitude and longitude of a point based on its relative normalization coordinates, the
        normalization factor and the latitude and longitude of the reference point.

        Parameters
        ----------
        position_ns_ew : torch.tensor
            contains the position of interest relative to a randomly chosen reference point.
        ref_lat_lon : tuple(float, float)
            latitude and longitude of the randomly chosen reference point.
        norm_factor : float
            normalization factor used to normalize the relative positions in km.
        """
        ref_lat, ref_lon = ref_lat_lon
        # denormalize the relative coordinates
        dist_north_south_km, dist_east_west_km = position_ns_ew * norm_factor
        # Calculate the absolute latitude that is dist_north_south_km km north (0°) or south (180°) of the reference point
        destination_lat = distance(kilometers=abs(dist_north_south_km)).destination((ref_lat, ref_lon),
                                                                                    0 if dist_north_south_km > 0 else 180).latitude
        # Calculate the absolute longitude that is dist_east_west_km km east (90°) or west (270°) of the reference point
        destination_lon = distance(kilometers=abs(dist_east_west_km)).destination((ref_lat, ref_lon),
                                                                                  90 if dist_east_west_km > 0 else 270).longitude
        return destination_lat, destination_lon

    def get_sample_by_source_id(self, source_id):
        """
        Returns a single sample from the Graph Instance Dataset based on the source ID according to the Istituto
        Nazionale di Geofisica e Vulcanologia (INGV).

        Parameters
        ----------
        source_id : int
            ID of the earthquake according to the INGV.

        Returns
        -------
        c.f. GIDataset.__getitem__
        """
        idx = int(np.where(np.array(self.source_id_list) == source_id)[0][0])
        return self[idx]

    def __getitem__(self, idx):
        """
        Returns a single sample from the Graph Instance Dataset.

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDataset)-1].

        Returns
        -------
        x : torch.tensor
            input time series as a tensor of shape [N,T,C], where N is the number of stations (differs between samples),
            T is the length of the time series, C is the number of channels per station.
        y : torch.tensor
            targets as a tensor of shape [N, 5], where N is the number of stations and 5 is the number of intensity
            measurements used as targets. These correspond to Peak Ground Acceleration PGA (cm/s^2), Peak Ground
            Velocity PGV (cm/s), and Peak Spectral Acceleration SA at 0.3s, 1.0s and 3.0s Periods (cm/s^2).
        L : torch.tensor
            normalized graph laplacian of shape [N, N] where N is the number of stations in the graph.
        source_df : pandas.DataFrame
            dataframe containing the metadata for the earthquake event.
        event_stations : list of str
            list of length N, where N is the number of stations in the graph, with the station names as strings
            (according to the International Registry of Seismograph Stations, IR), which recorded the event.
        channels : list of str
            list of length N, where N is the number of channels in the graph, with the channel names (as strings), which
            were used to derive the input time series x. The channel names identify the sampling and instrument gain
            (e.g. HN, HH, EH).
        """
        # get metadata and waveforms
        streams, y, source_df, event_stations, channels = self.get_streams_ims_by_idx(idx,
                                                                                      differentiate_velocities=self.differentiate_velocities)
        # trim the streams to the desired length of the time series
        streams, masked = self.trim_streams(streams, source_df)
        # convert the streams to PyTorch Tensors
        x = self.streams_to_tensor(streams)
        # construct the station network
        adj = self.construct_graph(source_df, event_stations)
        # get node and epicenter positions
        node_dists, epi_dists, norm_factor, ref_lat_lon = self.get_node_positions(source_df)
        # get node elevations
        node_elevations = self.get_node_elevations(source_df)
        return x, y, adj, source_df, event_stations, channels, node_dists, epi_dists, norm_factor, ref_lat_lon, node_elevations, masked


class GIDatasetPyG(GIDataset):
    def __getitem__(self, idx):
        """
        Returns a single sample from the Graph Instance Dataset compatible with Pytorch Geometric.

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDatasetPyG)-1].

        Returns
        -------
        data : torch_geometric.data.Data object
            contains all the data for a single sample, i.e.:
            - 'x' contains the flattened time series (e.g. 1000 values per time series over 3 components of motion ->
              3000 values); then the relative normalized node positions (2 values); then the node elevations (1 value).
            - 'y' contains the 5 intensity measurements for each station, which are the node-level targets.
            - 'graph_y' contains the relative normalized position of the earthquake epicenter (2 values), the used
              normalization factor (1 value), and the reference point latitude/longitude (2 values).
            - 'edge_index' contains pairs of indices of the nodes which are connected through an edge.
            - 'edge_attr' contains the weights associated with each of the edges.
            - 'batch' contains the indices of the graphs, i.e. which element in x, y and y_graph belongs to which graph.
        """
        # get sample
        x, y, adj, source_df, event_stations, channels, node_dists, epi_dists, norm_factor, ref_lat_lon, node_elevations, masked = super().__getitem__(
            idx)
        # convert masked array (which station was dropped out) to tensor
        masked = torch.tensor(masked, dtype=torch.bool)
        # convert adjacency matrix to edge_index and edge_attr
        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
        # Flatten time series inputs
        x = x.reshape(x.shape[0], -1)
        # concatenate with static node features
        x = torch.cat((x, node_dists, node_elevations.unsqueeze(1)), dim=1)
        # concatenate graph level features and information
        graph_y = torch.cat((epi_dists, norm_factor.unsqueeze(0), ref_lat_lon), dim=0)
        # create a torch_geometric.data.Data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_y=graph_y,
                                         masked=masked)
        return data


class GIDatasetPretrain(GIDataset):
    """
    Class for reading the Graph Instance Dataset for pretraining on the PGA.

    Parameters
    ----------
    meta_csv_path : str or pathlib.Path object
        path to the csv file containing the metadata.
    distances_npy_path : str or pathlib.Path object
        path to the npy file containing the distances between each pair of stations.
    h5_path : str or pathlib.Path object
        path to the hdf5 file containing the waveform data.
    split_path : str or pathlib.Path object
        path to the txt file containing the train, validation or test split.
    ts_length : int (Default: 10)
        length of the desired input time series in seconds.
    duration : int (Default: 20)
        length of the time series to derive the pretraining targets (PGA) from; should be longer than ts_length.
    edge_cutoff : float (Default: 0.0)
        threshold for cutting edges in the graph according to Bloemheuvel et al. 2023. Has to be in [0,1].
    norm : bool (Default: True)
        whether to norm the time series of a sample by the maximum value
    dropout : float (Default: 0.0)
        fraction of stations to mask out (except for the first one to receive the signal).
    graph_format : str (Default: 'complete')
        which type of graph to return for a sample. 'complete' does not filter out edges.
        'reduced' removes low-weighted edges as long as the graph stays a single component.
    min_node_degree : int (Default: None)
        minimum node degree in the graph if low-weighted edges are removed (i.e. graph='reduced).
    k_hop_reachability: int (Default: None)
        maximum diameter that the graph is allowed to have. E.g. 2 will guarantee that each node can reach each other
        node in the graph within 2 steps of message passing.
    pad_ts_s : float / int (Default: 0.0)
        If set to 0, the time series time window starts from the first P arrival time. Otherwise, it starts a
        random number of seconds [0, pad_ts_s] before the first P arrival time.
    deterministic : bool (Default: False)
        if set to True, the source_id is set as a random seed before drawing a random number in [0, pad_ts_s] to shift
        the time series. This means each sample is always returned the same way.
    verbose : bool (Default: False)
        If set to true, prints out information about the dataset such as data on the graph and plots the graph.

    Attributes
    ----------
    h5_path : str or pathlib.Path object
        path to the hdf5 file containing the waveform data.
    split_path : str or pathlib.Path object
        path to the txt file containing the train, validation or test split.
    meta_data_df : pandas.DataFrame
        contains the metadata of the dataset according to the documentation of the Instance dataset.
    source_id_list : list of str
        contains the IDs of all earthquake events in the dataset according to the Istituto Nazionale di Geofisica e
        Vulcanologia (INGV).
    station_id_list : list of str
        contains the IDs of all stations according to the International Registry of Seismograph Stations, IR.
    distances : numpy.ndarray
        contains the distances in km between each pair of stations.
    edge_cutoff : float (Default: 0.0)
        threshold for cutting edges in the graph according to Bloemheuvel et al. 2023. Has to be in [0,1].
    ts_length : int (Default: 10)
        length of the desired input time series in seconds.
    duration : int (Default: 20)
        length of the time series to derive the pretraining targets (PGA) from; should be longer than ts_length.
    frequency : int or float (Default: 100)
        sampling frequency in Hz.
    n_channels : int (Default: 3)
        number of channels per recording (E:east-west, N:north-south, Z:up-down).
    pad : bool (Default: True)
        whether to pad the time series to the desired length if needed.
    fill_value : float (Default: 0)
        value to fill missing values with during padding.
    differentiate_velocities : bool (Default: True)
        whether to differentiate velocities to accelerations for HH and EH channels.
    norm : bool (Default: True)
        whether to norm the time series of a sample by the maximum value
    dropout : float (Default: 0.0)
        fraction of stations to mask out (except for the first one to receive the signal).
    graph_format : str (Default: 'complete')
        which type of graph to return for a sample. 'complete' does not filter out edges.
        'reduced' removes low-weighted edges as long as the graph stays a single component.
    min_node_degree : int (Default: None)
        minimum node degree in the graph if low-weighted edges are removed (i.e. graph='reduced).
    k_hop_reachability: int (Default: None)
        maximum diameter that the graph is allowed to have. E.g. 2 will guarantee that each node can reach each other
        node in the graph within 2 steps of message passing.
    pad_ts_s : float / int (Default: 0.0)
        If set to 0, the time series time window starts from the first P arrival time. Otherwise, it starts a
        random number of seconds [0, pad_ts_s] before the first P arrival time.
    deterministic : bool (Default: False)
        if set to True, the source_id is set as a random seed before drawing a random number in [0, pad_ts_s] to shift
        the time series. This means each sample is always returned the same way.
    verbose : bool (Default: False)
        If set to true, prints out information about the dataset such as data on the graph and plots the graph.
    """

    def __init__(self, meta_csv_path, distances_npy_path, h5_path, split_path=None, ts_length=10, duration=20,
                 edge_cutoff=0.0, norm=True, dropout=0.0, graph_format='complete', min_node_degree=None,
                 k_hop_reachability=None, pad_ts_s=0, deterministic=False, verbose=False):
        self.h5_path = h5_path
        self.split_path = split_path
        self.meta_data_df = self.get_meta_data(meta_csv_path)
        self.source_id_list = sorted(self.meta_data_df['source_id'].unique())
        self.filter_for_split()  # if a split is given, filter out all other events not contained in the split
        self.station_id_list = sorted(self.meta_data_df['station_code'].unique())
        self.distances = np.load(distances_npy_path)
        self.edge_cutoff = edge_cutoff  # cut edges with lower weight in the graph
        self.ts_length = ts_length  # length of time series in seconds
        self.duration = duration  # length of the time series in seconds from which the targets (PGA) are derived
        self.frequency = 100  # sampling frequency in Hz
        self.n_channels = 3  # number of channels per recording: 3 (E:east-west, N:north-south, Z:up-down)
        self.pad = True  # pad time series which are not complete within
        self.fill_value = 0  # value to pad shorter time series with
        self.differentiate_velocities = True  # whether to differentiate velocities to accelerations
        self.norm = norm  # whether to norm the time series of a sample
        self.dropout = dropout  # fraction of stations to mask out (except for the first station to receive the signal)
        self.graph_format = graph_format  # which type of graph to use (complete, reduced)
        self.min_node_degree = min_node_degree  # minimum node degree for the graph
        self.k_hop_reachability = k_hop_reachability  # maximum graph diameter
        self.pad_ts_s = pad_ts_s  # pad the time series start with up to pad_ts_s seconds
        self.deterministic = deterministic  # whether to make the sampling deterministic or not
        self.verbose = verbose  # whether to print infos e.g. about the graph or plot the graph structure

    def trim_streams(self, streams, source_df):
        """
        Trim a list of streams to the desired time series length, starting at earthquake origin time.

        Parameters
        ----------
        streams : list of obspy.core.stream.Stream objects
            List of all stream objects for the event. Each stream corresponds to one station. Each trace in a stream
            correspond to one channel (East-West E, North-South N, Up-Down Z).
        source_df : pandas.DataFrame
            DataFrame containing the event metadata, including the origin time of the earthquake.

        Returns
        -------
        trimmed_streams : list of obspy.core.stream.Stream objects
            Contains the streams trimmed to the desired length of the time series, starting at the origin time of the
            earthquake. Shorter time series are padded if desired.
        """

        # determine earliest arrival time for the P(rimary) wave
        p_arrival_times = [oc.UTCDateTime(x) for x in
                           source_df['trace_P_arrival_time']]  # use trace_P_arrival_time or source_origin_time
        earliest_p_arrival_time = min(p_arrival_times)
        latest_p_arrival_time = max(p_arrival_times)

        # -------------------Trimming for Pretraining----------------------
        # Compute average time when the time series end
        if self.deterministic:
            seed = int(source_df['source_id'].iloc[0])
            random.seed(seed)
        starttime = oc.UTCDateTime(random.uniform(earliest_p_arrival_time.timestamp, latest_p_arrival_time.timestamp))
        # randomly set the start of the timeseries a few seconds earlier
        if self.deterministic:
            seed = int(source_df['source_id'].iloc[0])
            random.seed(seed)
        pad_ts = random.uniform(0, self.pad_ts_s)
        self._print(
            f'Padding of the time series: {pad_ts}s. Earliest p-arrival-time: {earliest_p_arrival_time}. Latest p-arrival-time: {latest_p_arrival_time}. start time: {starttime}')
        starttime = starttime - pad_ts

        # obtain TARGETS from longer time series
        stoptime = starttime + self.duration
        # trim the time series to the [starttime, stoptime] interval
        y = []
        for stream in streams:
            # trim the stream to the desired duration
            trimmed_stream = stream.copy().trim(starttime, stoptime, pad=self.pad, fill_value=self.fill_value)
            # get the maximum acceleration from this longer time window (Peak Ground Acceleration)
            pga = max(abs(trace.data).max() for trace in trimmed_stream)
            y.append(pga)
        # convert it to pytorch tensors
        y = torch.tensor(y).unsqueeze(1)
        # determine targets which are 0 (should be excluded from the optimization)
        non_zero_mask = y > 0

        # -------------------Masking out Stations----------------------
        if self.deterministic:
            seed = int(source_df['source_id'].iloc[0])
            random.seed(seed)
        # Masking needs to happen after obtaining the targets, otherwise we mask out the target (PGA) too.
        # Find the index of the first station to receive the signal
        first_station_idx = p_arrival_times.index(min(p_arrival_times))
        # Get indices of the remaining stations
        remaining_indices = [i for i in range(len(streams)) if i != first_station_idx]
        # Randomly select 20% of the remaining stations
        num_to_mask = int(len(streams) * self.dropout)
        mask_indices = random.sample(remaining_indices, min(num_to_mask, len(remaining_indices)))
        # set selected streams to 0
        masked = np.zeros(len(streams))
        for idx in mask_indices:
            for trace in streams[idx]:
                trace.data[:] = 0.
                pass
            # array to keep track which streams were masked
            masked[idx] = 1.

        # Obtain INPUT time series by trimming to self.ts_length
        stoptime = starttime + self.ts_length
        # trim the time series to the [starttime, stoptime] interval
        trimmed_streams = []
        for stream in streams:
            trimmed_stream = stream.copy().trim(starttime, stoptime, pad=self.pad, fill_value=self.fill_value)
            trimmed_streams.append(trimmed_stream)

        return trimmed_streams, y, masked, non_zero_mask

    def __getitem__(self, idx):
        """
        Returns a single sample from the Graph Instance Dataset.
        The returned sample is for pretraining purposes, cut to a random section of the whole time
        series. The target is the PGA from the same cut time series + some additional duration; e.g.
        the input time series + 10s; such that the model cannot cheat.

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDataset)-1].

        Returns
        -------
        x : torch.tensor
            input time series as a tensor of shape [N,T,C], where N is the number of stations (differs between samples),
            T is the length of the time series, C is the number of channels per station.
        y : torch.tensor
            targets as a tensor of shape [N, 5], where N is the number of stations and 5 is the number of intensity
            measurements used as targets. These correspond to Peak Ground Acceleration PGA (cm/s^2), Peak Ground
            Velocity PGV (cm/s), and Peak Spectral Acceleration SA at 0.3s, 1.0s and 3.0s Periods (cm/s^2).
        L : torch.tensor
            normalized graph laplacian of shape [N, N] where N is the number of stations in the graph.
        source_df : pandas.DataFrame
            dataframe containing the metadata for the earthquake event.
        event_stations : list of str
            list of length N, where N is the number of stations in the graph, with the station names as strings
            (according to the International Registry of Seismograph Stations, IR), which recorded the event.
        channels : list of str
            list of length N, where N is the number of channels in the graph, with the channel names (as strings), which
            were used to derive the input time series x. The channel names identify the sampling and instrument gain
            (e.g. HN, HH, EH).
        """
        # get metadata and waveforms
        streams, y, source_df, event_stations, channels = self.get_streams_ims_by_idx(idx,
                                                                                      differentiate_velocities=self.differentiate_velocities)
        # trim the streams to the desired length of the time series and obtain the new targets
        streams, y, masked, non_zero_mask = self.trim_streams(streams, source_df)
        # convert the streams to PyTorch Tensors
        x = self.streams_to_tensor(streams)
        # construct the station network
        adj = self.construct_graph(source_df, event_stations)
        # get node and epicenter positions
        node_dists, epi_dists, norm_factor, ref_lat_lon = self.get_node_positions(source_df)
        # get node elevations
        node_elevations = self.get_node_elevations(source_df)
        return x, y, adj, source_df, event_stations, channels, node_dists, epi_dists, norm_factor, ref_lat_lon, node_elevations, masked, non_zero_mask


class GIDatasetPretrainPyG(GIDatasetPretrain):
    def __getitem__(self, idx):
        """
        Returns a single sample from the Graph Instance Dataset compatible with Pytorch Geometric.
        The returned sample is for pretraining purposes, cut to a random section of the whole time
        series. The target is the PGA from the same cut time series + some additional duration; e.g.
        the input time series + 10s; such that the model cannot cheat.

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDatasetPyG)-1].

        Returns
        -------
        data : torch_geometric.data.Data object
            contains all the data for a single sample, i.e.:
            - 'x' contains the flattened time series (e.g. 1000 values per time series over 3 components of motion ->
              3000 values); then the relative normalized node positions (2 values); then the node elevations (1 value).
            - 'y' contains the 5 intensity measurements for each station, which are the node-level targets.
            - 'graph_y' contains the relative normalized position of the earthquake epicenter (2 values), the used
              normalization factor (1 value), and the reference point latitude/longitude (2 values).
            - 'edge_index' contains pairs of indices of the nodes which are connected through an edge.
            - 'edge_attr' contains the weights associated with each of the edges.
            - 'batch' contains the indices of the graphs, i.e. which element in x, y and y_graph belongs to which graph.
        """
        # get sample
        x, y, adj, source_df, event_stations, channels, node_dists, epi_dists, norm_factor, ref_lat_lon, node_elevations, masked, non_zero_mask = super().__getitem__(
            idx)
        # convert masked array (which station was dropped out) to tensor
        masked = torch.tensor(masked, dtype=torch.bool)
        # convert adjacency matrix to edge_index and edge_attr
        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
        # Flatten time series inputs
        x = x.reshape(x.shape[0], -1)
        # concatenate with static node features
        x = torch.cat((x, node_dists, node_elevations.unsqueeze(1)), dim=1)
        # concatenate graph level features and information
        graph_y = torch.cat((epi_dists, norm_factor.unsqueeze(0), ref_lat_lon), dim=0)
        # create a torch_geometric.data.Data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_y=graph_y,
                                         masked=masked, non_zero_mask=non_zero_mask)
        return data


class GIDatasetPyG_with_artificial_node(GIDataset):
    def __init__(self, meta_csv_path, distances_npy_path, h5_path, split_path=None, ts_length=10, edge_cutoff=0.0,
                 norm=True, dropout=0.0, graph_format='complete', min_node_degree=None, k_hop_reachability=None,
                 pad_ts_s=0, deterministic=False, random_ts_padding=True, filter_artificial_node_edges=True,
                 verbose=False):
        self.h5_path = h5_path
        self.split_path = split_path
        self.meta_data_df = self.get_meta_data(meta_csv_path)
        self.source_id_list = sorted(self.meta_data_df['source_id'].unique())
        self.filter_for_split()  # if a split is given, filter out all other events not contained in the split
        self.station_id_list = sorted(self.meta_data_df['station_code'].unique())
        self.distances = np.load(distances_npy_path)
        self.edge_cutoff = edge_cutoff  # cut edges with lower weight in the graph
        self.ts_length = ts_length  # length of time series in seconds
        self.frequency = 100  # sampling frequency in Hz
        self.n_channels = 3  # number of channels per recording: 3 (E:east-west, N:north-south, Z:up-down)
        self.pad = True  # pad time series which are not complete within
        self.fill_value = 0  # value to pad shorter time series with
        self.differentiate_velocities = True  # whether to differentiate velocities to accelerations
        self.norm = norm  # whether to norm the time series of a sample
        self.dropout = dropout  # ratio of stations to mask out
        self.graph_format = graph_format  # which type of graph to use (complete, reduced)
        self.min_node_degree = min_node_degree  # minimum node degree for the graph
        self.k_hop_reachability = k_hop_reachability  # maximum graph diameter
        self.pad_ts_s = pad_ts_s  # pad the time series start with up to pad_ts_s seconds
        self.deterministic = deterministic  # whether the time series sampling should be deterministic
        self.random_ts_padding = random_ts_padding  # whether the time series padding should be random
        self.verbose = verbose  # whether to print infos e.g. about the graph or plot the graph structure

        self.graph = None  # graph without artificial node, e.g. for plotting
        self.graph_with_artificial_node = None  # graph with artificial node, e.g. for plotting
        self.filter_artificial_node_edges = filter_artificial_node_edges

    def __getitem__(self, idx):
        """
        Returns a single sample from the Graph Instance Dataset compatible with Pytorch Geometric.

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDatasetPyG)-1].

        Returns
        -------
        data : torch_geometric.data.Data object
            contains all the data for a single sample, i.e.:
            - 'x' contains the flattened time series (e.g. 1000 values per time series over 3 components of motion ->
              3000 values); then the relative normalized node positions (2 values); then the node elevations (1 value).
            - 'y' contains the 5 intensity measurements for each station, which are the node-level targets.
            - 'graph_y' contains the relative normalized position of the earthquake epicenter (2 values), the used
              normalization factor (1 value), and the reference point latitude/longitude (2 values).
            - 'edge_index' contains pairs of indices of the nodes which are connected through an edge.
            - 'edge_attr' contains the weights associated with each of the edges.
            - 'batch' contains the indices of the graphs, i.e. which element in x, y and y_graph belongs to which graph.
        """
        # get sample
        x, y, adj, source_df, event_stations, channels, node_dists, epi_dists, norm_factor, ref_lat_lon, node_elevations, masked = super().__getitem__(
            idx)
        # convert masked array (which station was dropped out) to tensor
        masked = torch.tensor(masked, dtype=torch.bool)
        # convert adjacency matrix to edge_index and edge_attr
        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
        # Flatten time series inputs
        x = x.reshape(x.shape[0], -1)
        # concatenate with static node features
        x = torch.cat((x, node_dists, node_elevations.unsqueeze(1)), dim=1)
        # concatenate graph level features and information
        graph_y = torch.cat((epi_dists, norm_factor.unsqueeze(0), ref_lat_lon), dim=0)
        # create a torch_geometric.data.Data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_y=graph_y,
                                         masked=masked)
        return data

    def construct_graph_with_artificial_node(self, source_df, event_stations, artificial_node_lat, artificial_node_lon,
                                             export_graph_path=None):
        """
        Construct the graph for a single event including a single artificial node.

        Parameters
        ----------
        source_df : pandas.DataFrame
            dataframe containing the metadata for a single earthquake event.
        event_stations : list of str
            list of length N, where N is the number of stations in the graph, with the station names as strings
            (according to the International Registry of Seismograph Stations, IR), which recorded the event.
        export_graph_path : str or pathlib.Path or None
            if not None, the edges of the graph will be exported to the given path as a csv file containing the edge
            endpoints as WKT LINESTRING that can be interpreted in GIS to visualize the graph.
        artificial_node_lat : float
            latitude of the artificial node to be inserted in the graph.
        artificial_node_lon : float
            longitude of the artificial node to be inserted in the graph.

        Returns
        -------
        L : torch.tensor
            normalized graph laplacian for the sample.
        """
        # Adapted from https://github.com/StefanBloemheuvel/GCNTimeseriesRegression/blob/main/graph_maker.py
        graph = nx.Graph()

        # Add nodes (stations) to the graph
        for station_code in event_stations:
            row = source_df[source_df['station_code'] == station_code].iloc[0]
            graph.add_node(row['station_code'], pos=(row['station_longitude_deg'], row['station_latitude_deg']))

        # Add edges (with geodesic distance) to the graph
        for s1 in event_stations:
            for s2 in event_stations:
                # find the correct entry in the distance matrix
                if s1 == s2:  # filters out self-loops
                    continue
                i1 = self.station_id_list.index(s1)
                i2 = self.station_id_list.index(s2)
                distance = self.distances[i1, i2]
                graph.add_edge(s1, s2, weight=distance, added_info=distance)

        # Make a new graph with weights higher for stations that are close to each other
        edge_list = nx.to_pandas_edgelist(graph)
        # normalize the distances
        edge_min = min(edge_list['weight'])
        edge_max = max(edge_list['weight'])
        edge_list['weight'] = (edge_list['weight'] - edge_min) / (
                edge_max - edge_min)
        # Large distances -> small weights
        edge_list['weight'] = 1.0 - edge_list[
            'weight']  # Bloemheuvel used 0.98 (in GitHub) instead of 1.0 (as in their paper)
        # Make adjacency matrix
        adj = nx.from_pandas_edgelist(edge_list, edge_attr=['weight'])
        adj = pd.DataFrame(nx.adjacency_matrix(adj, weight='weight').todense())
        # apply threshold
        adj[adj < self.edge_cutoff] = 0
        # create new graph with updated edge weights
        newgraph = nx.from_pandas_adjacency(adj)
        # relabel the nodes
        name_dict = dict(zip([x for x in range(len(event_stations))], [x for x in graph.nodes()]))
        newgraph = nx.relabel_nodes(newgraph, name_dict)
        # add longitude/latitude to new graph nodes
        nx.set_node_attributes(newgraph, nx.get_node_attributes(graph, 'pos'), 'pos')
        # add distance in km to new graph edges
        nx.set_edge_attributes(newgraph, nx.get_edge_attributes(graph, 'added_info'), 'added_info')

        self._print(newgraph)

        # reduce graph if desired
        if self.graph_format == 'reduced':
            edges = list(newgraph.edges())
            # sort edges by their weights
            edges = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'])

            for edge in edges:
                # print(u, v)
                u, v, attributes = edge
                weight = attributes['weight']
                added_info = attributes['added_info']
                # remove the edge
                newgraph.remove_edge(u, v)
                # STOPPING CRITERIA
                # 1) check if graph is no longer connected
                if not nx.is_connected(newgraph):
                    # add the last edge back to keep the graph connected
                    newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                    self._print('Edge added back in (Splitting of Component): ', u, v, newgraph[u][v])
                    break
                # 2) check if each node still has the minimum node degree
                if self.min_node_degree is not None:
                    if newgraph.degree[u] < self.min_node_degree or newgraph.degree[v] < self.min_node_degree:
                        # add the last edge back to maintain the minimum edge degree
                        newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                        self._print('Edge added back in (Min Node Degree): ', u, v, newgraph[u][v])
                        break
                # 3) check if the graph still has a diameter below the desired limit
                if self.k_hop_reachability is not None:
                    if nx.diameter(newgraph) > self.k_hop_reachability:
                        # add the last edge back to keep the graph diameter
                        newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                        self._print('Edge added back in (k-hop reachability): ', u, v, newgraph[u][v])
                        break

        elif self.graph_format == 'complete':
            pass
        else:
            raise ValueError('Graph format not recognized')
        self._print(newgraph)

        self.graph = newgraph.copy()

        # add artificial node
        newgraph.add_node('node', pos=(artificial_node_lon, artificial_node_lat))
        # go through all nodes and add edges to the artificial node
        for node, data in newgraph.nodes(data=True):
            # don't add a self-loop
            if node == 'node':
                continue
            # print(node, data)
            # get position of the node
            node_lon, node_lat = data['pos']
            # calculate distance to the artificial node
            distance = geodesic((node_lat, node_lon), (artificial_node_lat, artificial_node_lon)).kilometers
            # normalize the distance using the old normalization
            weight = (distance - edge_min) / (edge_max - edge_min)
            # if the distance is shorter
            # Large distances -> small weights
            weight = 1.0 - weight
            # handle negative weights (edges exceeding the previously longest distance between two edges) by clamping
            # to a small non-negative value.
            weight = max(0.01, weight)
            # add edge to the graph
            newgraph.add_edge(node, 'node', weight=weight, added_info=distance)
        # Filter out newly added edges connecting to the artificial node
        if not self.filter_artificial_node_edges:
            pass
        elif self.graph_format == 'reduced':
            edges = list(newgraph.edges())
            # sort edges by their weights
            edges = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'])

            for edge in edges:
                # print(u, v)
                u, v, attributes = edge
                # ======================================================================================
                # check whether edge contains artificial node, if not, skip the filtering for this edge
                # this maintains the edges of the original graph
                if u != 'node' and v != 'node':
                    continue
                # ======================================================================================
                weight = attributes['weight']
                added_info = attributes['added_info']
                # remove the edge
                newgraph.remove_edge(u, v)
                # STOPPING CRITERIA
                # 1) check if graph is no longer connected
                if not nx.is_connected(newgraph):
                    # add the last edge back to keep the graph connected
                    newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                    self._print('Edge added back in (Splitting of Component): ', u, v, newgraph[u][v])
                    break
                # 2) check if each node still has the minimum node degree
                if self.min_node_degree is not None:
                    if newgraph.degree[u] < self.min_node_degree or newgraph.degree[v] < self.min_node_degree:
                        # add the last edge back to maintain the minimum edge degree
                        newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                        self._print('Edge added back in (Min Node Degree): ', u, v, newgraph[u][v])
                        break
                # 3) check if the graph still has a diameter below the desired limit
                if self.k_hop_reachability is not None:
                    if nx.diameter(newgraph) > self.k_hop_reachability:
                        # add the last edge back to keep the graph diameter
                        newgraph.add_edge(u, v, weight=weight, added_info=added_info)
                        self._print('Edge added back in (k-hop reachability): ', u, v, newgraph[u][v])
                        break

        elif self.graph_format == 'complete':
            pass
        else:
            raise ValueError('Graph format not recognized')
        self._print(newgraph)

        self.graph_with_artificial_node = newgraph.copy()

        # Export the graph as a csv file if desired
        if export_graph_path:
            self.graph_to_wkt(newgraph, source_df, export_graph_path)

        # --------------------------------------------------------------------------------------------------------------
        # print some graph info
        edges1 = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'], reverse=False)[0][0]
        edges2 = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'], reverse=False)[0][1]
        edges3 = sorted(newgraph.edges(data=True), key=lambda x: x[2]['weight'], reverse=False)[0][2]
        self._print('---------------------------------------------------------------------------------')
        self._print(f'edge with lowest weight = {edges1, edges2, edges3}, OG km = {newgraph[edges1][edges2]}')
        self._print(
            f'Average degree of the graph =  {np.mean([val for (node, val) in sorted(newgraph.degree(), key=lambda pair: pair[0])])}')

        degree_centralities = []
        for i in nx.degree_centrality(newgraph).values():
            degree_centralities.append(i)
        self._print('avg degree centrality = ', np.mean(degree_centralities))

        distances_og = []
        for i in newgraph.edges(data=True):
            distances_og.append(i[2]['added_info'])
        self._print('average distance og = ', np.array(distances_og).mean())

        # plot graph
        if self.verbose:
            plot_weights = [newgraph[u][v]['weight'] for u, v in newgraph.edges()]
            nx.draw(newgraph, nx.get_node_attributes(newgraph, 'pos'), with_labels=True, width=plot_weights)
            plt.show()
        # --------------------------------------------------------------------------------------------------------------

        # Make adjacency matrix of the filtered graph
        new_edge_list = nx.to_pandas_edgelist(newgraph)
        new_adj = nx.from_pandas_edgelist(new_edge_list, edge_attr=['weight'])
        new_adj = pd.DataFrame(nx.adjacency_matrix(new_adj, weight='weight').todense())

        new_adj = new_adj.to_numpy()
        new_adj = torch.from_numpy(new_adj)
        return new_adj

    def get_node_positions_with_artificial_node(self, source_df, artificial_node_lat, artificial_node_lon):
        """
        Get relative, normalized positions of the nodes in the graph and the earthquake epicenter relative to a random
        reference point.

        Parameters
        ----------
        source_df : pandas.DataFrame
            dataframe containing the metadata of the sample, including station and earthquake positions.

        Returns
        -------
        node_dists : torch.tensor
            contains the positions of each node in the graph relative to a randomly chosen reference point within
            min/max of station latitudes and longitudes. The relative positions in km in north/south and east/west
            direction are normalized by the maximum absolute distance component.
        epi_dists : torch.tensor
            same as node_dists, but for the normalized relative position of the earthquake epicenter.
        norm_factor : float
            normalization factor used to normalize the relative positions in km.
        ref_lat_lon : tuple(float, float)
            latitude and longitude of the randomly chosen reference point.
        """
        # filter source_df for unique stations
        source_df = source_df.drop_duplicates(subset=['station_code'])
        # get latitudes and longitudes of stations and epicenter
        lats = torch.tensor(source_df['station_latitude_deg'].values, dtype=torch.float32)
        lons = torch.tensor(source_df['station_longitude_deg'].values, dtype=torch.float32)
        epi_lat = torch.tensor(source_df['source_latitude_deg'].values[0], dtype=torch.float32)
        epi_lon = torch.tensor(source_df['source_longitude_deg'].values[0], dtype=torch.float32)
        self._print(f'Min/Max Node Latitude: {lats.min()}/{lats.max()}; Longitude: {lons.min()}/{lons.max()}')
        # get random reference point
        ref_lat = random.uniform(min(lats), max(lats))
        ref_lon = random.uniform(min(lons), max(lons))
        ref_lat_lon = torch.tensor((ref_lat, ref_lon), dtype=torch.float32)

        # Add artificial node coordinates
        lats = torch.cat((lats, torch.tensor(artificial_node_lat).unsqueeze(0)), dim=0)
        lons = torch.cat((lons, torch.tensor(artificial_node_lon).unsqueeze(0)), dim=0)

        # get distances in x (lon) and y (lat) direction between nodes and reference point
        def get_dists(ref_lat, ref_lon, lat, lon):
            # calculate distance along the latitude (North-South)
            dist_north_south_km = geodesic((ref_lat, ref_lon), (lat, ref_lon)).km
            # calculate distance along the longitudes (East-West)
            dist_east_west_km = geodesic((ref_lat, ref_lon), (ref_lat, lon)).km
            # determine the sign based on the relative position
            if lat < ref_lat:  # if node is further south (smaller lat) than reference point
                dist_north_south_km *= -1
            if lon < ref_lon:  # if node is further west (smaller lon) than reference point
                dist_east_west_km *= -1
            return dist_north_south_km, dist_east_west_km

        node_dists = torch.tensor([list(get_dists(ref_lat, ref_lon, lat, lon)) for lat, lon in zip(lats, lons)],
                                  dtype=torch.float32)
        # get distance between epicenter and reference point
        epi_dists = torch.tensor(get_dists(ref_lat, ref_lon, epi_lat, epi_lon), dtype=torch.float32)
        # get normalization factor as max distance component in km
        # exclude the artificially added node from this
        norm_factor = torch.tensor(node_dists[:-1].abs().max(), dtype=torch.float32)
        # normalize distances
        node_dists = node_dists / norm_factor
        epi_dists = epi_dists / norm_factor
        return node_dists, epi_dists, norm_factor, ref_lat_lon

    def get_node_elevations_with_artificial_node(self, source_df, artificial_node_lat, artificial_node_lon):
        """
        Get the normalized elevations of the nodes in the graph.

        Parameters
        ----------
        source_df : pandas.DataFrame
            dataframe with the sample metadata including the elevations.

        Returns
        -------
        elevations : torch.tensor
            elevations of the nodes in the graph.
        """
        # filter source_df for unique stations
        source_df = source_df.drop_duplicates(subset=['station_code'])
        # get elevations in m
        elevations = torch.tensor(source_df['station_elevation_m'].values, dtype=torch.float32)
        elevations_max = elevations.max()

        # get elevation for the artificial node
        def get_elevation(lat, lon):
            query = (f'https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}')
            r = requests.get(query).json()
            elevation = r['elevation'][0]
            return elevation

        artificial_node_elevation = get_elevation(lat=artificial_node_lat, lon=artificial_node_lon)
        # append result with the original node elevations
        elevations = torch.cat((elevations, torch.tensor(artificial_node_elevation).unsqueeze(0)), dim=0)
        elevations /= elevations_max
        return elevations

    def get_item_with_artificial_node(self, idx, artificial_node_lat, artificial_node_lon):
        """
        Returns a single sample from the Graph Instance Dataset compatible with Pytorch Geometric, including a single
        artificial node inserted into the graph. The artificial node inserted into the graph will only be added to the
        input x (as tensor filled with 0s) and to the graph (edge_index, edge_attr). The

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDatasetPyG)-1].
        artificial_node_lat : float
            latitude of the artificial node to be inserted in the graph.
        artificial_node_lon : float
            longitude of the artificial node to be inserted in the graph.

        Returns
        -------
        data : torch_geometric.data.Data object
            contains all the data for a single sample, i.e.:
            - 'x' contains the flattened time series (e.g. 1000 values per time series over 3 components of motion ->
              3000 values); then the relative normalized node positions (2 values); then the node elevations (1 value).
            - 'y' contains the 5 intensity measurements for each station, which are the node-level targets.
            - 'graph_y' contains the relative normalized position of the earthquake epicenter (2 values), the used
              normalization factor (1 value), and the reference point latitude/longitude (2 values).
            - 'edge_index' contains pairs of indices of the nodes which are connected through an edge.
            - 'edge_attr' contains the weights associated with each of the edges.
            - 'batch' contains the indices of the graphs, i.e. which element in x, y and y_graph belongs to which graph.
        """
        # get metadata and waveforms
        streams, y, source_df, event_stations, channels = self.get_streams_ims_by_idx(idx,
                                                                                      differentiate_velocities=self.differentiate_velocities)
        # trim the streams to the desired length of the time series
        streams, masked = self.trim_streams(streams, source_df)
        # convert the streams to PyTorch Tensors
        x = self.streams_to_tensor(streams)
        # add empty time series for the artificial node
        artificial_node_input = torch.zeros_like(x[-1]).unsqueeze(0)
        x = torch.cat((x, artificial_node_input), dim=0)
        # construct the station network
        adj = self.construct_graph_with_artificial_node(source_df, event_stations, artificial_node_lat,
                                                        artificial_node_lon)
        # get node and epicenter positions
        node_dists, epi_dists, norm_factor, ref_lat_lon = self.get_node_positions_with_artificial_node(source_df,
                                                                                                       artificial_node_lat,
                                                                                                       artificial_node_lon)
        # get node elevations
        node_elevations = self.get_node_elevations_with_artificial_node(source_df, artificial_node_lat,
                                                                        artificial_node_lon)

        # convert masked array (which station was dropped out) to tensor
        masked = torch.tensor(masked, dtype=torch.bool)
        # convert adjacency matrix to edge_index and edge_attr
        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
        # Flatten time series inputs
        x = x.reshape(x.shape[0], -1)
        # concatenate with static node features
        x = torch.cat((x, node_dists, node_elevations.unsqueeze(1)), dim=1)
        # concatenate graph level features and information
        graph_y = torch.cat((epi_dists, norm_factor.unsqueeze(0), ref_lat_lon), dim=0)
        # create a torch_geometric.data.Data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_y=graph_y,
                                         masked=masked)
        return data


class ArtificialNodeIterator:
    def __init__(self, idx, pyg_dataset, grid_size=7):
        """
        Iterator that returns a sample from the Graph Instance dataset, with artificial nodes inserted into the graph.

        Parameters
        ----------
        idx : int
            index of the sample to return. Accepts values in [0, len(GIDatasetPyG)-1].
        pyg_dataset : GIDatasetPyG_with_artificial_node
            torch geometric dataset object which allows inserting a single artificial node into the graph.
        grid_size : int
            number of grid points in NS and EW direction.
        """
        self.idx = idx
        self.pyg_dataset = pyg_dataset
        self.pyg_dataset.verbose = False  # deactivate verbose output
        self.grid_size = grid_size
        self.index = 0
        self.source_df = None
        self.grid_coords = self.get_grid_coords()

    def get_grid_coords(self):
        x, y, adj, source_df, event_stations, channels, node_dists, epi_dists, norm_factor, ref_lat_lon, node_elevations, masked = super(
            type(self.pyg_dataset), self.pyg_dataset).__getitem__(self.idx)
        self.source_df = source_df
        # get min/max lat/lon
        min_lat, max_lat = source_df['station_latitude_deg'].min(), source_df['station_latitude_deg'].max()
        min_lon, max_lon = source_df['station_longitude_deg'].min(), source_df['station_longitude_deg'].max()
        # compute ranges
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        # determine the step size using the larger range
        step_size = max(lat_range, lon_range) / (self.grid_size - 1)
        # generate grid points
        if lat_range >= lon_range:
            lat_points = np.linspace(min_lat, min_lat + step_size * (self.grid_size - 1), self.grid_size)
            lon_points = np.linspace(min_lon - (lat_range - lon_range) / 2,
                                     min_lon - (lat_range - lon_range) / 2 + step_size * (self.grid_size - 1),
                                     self.grid_size)
        else:
            lat_points = np.linspace(min_lat - (lon_range - lat_range) / 2,
                                     min_lat - (lon_range - lat_range) / 2 + step_size * (self.grid_size - 1),
                                     self.grid_size)
            lon_points = np.linspace(min_lon, min_lon + step_size * (self.grid_size - 1), self.grid_size)
        # create grid of coordinate pairs (grid in lat-lon space)
        grid_coords = np.array([(lat, lon) for lat in lat_points for lon in lon_points])
        return grid_coords

    def __len__(self):
        return len(self.grid_coords)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.grid_coords):
            raise StopIteration
        lat, lon = self.grid_coords[self.index]
        sample = self.pyg_dataset.get_item_with_artificial_node(self.idx, lat, lon)
        self.index += 1
        return lat, lon, sample
