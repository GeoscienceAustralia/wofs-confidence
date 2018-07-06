import yaml
import pickle
from datacube.api import GridWorkflow
from datacube.model import GridSpec, Variable
from datacube.utils.geometry import GeoBox, Coordinate, CRS
from datacube.storage import netcdf_writer
from datacube.storage.storage import create_netcdf_storage_unit
from datacube import Datacube
import numpy as np
from datetime import datetime
from pathlib import Path
import logging


DEFAULT_CRS = 'EPSG:3577'
DEFAULT_NODATA = np.nan
DEFAULT_TYPE = 'float32'
DEFAULT_THRESHOLD_FILTERED = 0.10


class TrainingModel(object):
    def __init__(self):
        pass


class Config(object):
    def __init__(self, config_file, metadata_template_file=None):
        with open(config_file, 'r') as cfg_stream:
            self.cfg = yaml.load(cfg_stream)
        # ToDo: metadata
        self.metadata_template = metadata_template_file
        # with open(metadata_template_file, 'r') as template_stream:
        #     self.metadata_template = yaml.load(template_stream)

    def get_env_of_product(self, product_name):
        for dc in self.cfg['datacubes']:
            for dc_env in dc.keys():
                if product_name in dc[dc_env]:
                    return dc_env

    def get_factor_info(self, factor_name):
        for factor in self.cfg['factors']:
            if factor_name == factor['name']:
                if factor.get('flag'):
                    return {'name': factor_name,
                            'env': self.get_env_of_product(factor['product']),
                            'product': factor['product'],
                            'band': factor['band'],
                            'flag': factor['flag']}
                else:
                    return {'name': factor_name,
                            'env': self.get_env_of_product(factor['product']),
                            'product': factor['product'],
                            'band': factor['band']}

    def get_confidence_model(self):
        """
            The model representation we use are sklearn.LogisticRegression objects with
            added 'factor' specifications, i.e. if model is a sklearn linear_model object,
            model.factors will have factor specifications. A factor is a dict object
            that hold information which is required to access factor data, typically, from
            a datacube.
        """
        model_path = self.cfg['confidence_filtering']['trained_model_path']
        
        # read the trained model
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class WofsFiltered(object):
    def __init__(self, config: Config, grid_spec: GridSpec):
        self.cfg = config
        self.grid_spec = grid_spec

    def load_tile_data(self, cell_index, factors):

        # Start: Mock code
        with Datacube(app='confidence_layer', env='prod') as dc:
            # Get the tile spec
            gwf = GridWorkflow(dc.index, self.grid_spec)
            indexed_tiles = gwf.list_cells(cell_index, product='ls5_nbar_albers')
            # load the data of the tile
            dataset = gwf.load(tile=indexed_tiles[cell_index], measurements=['blue'])
            mock_data = dataset.data_vars['blue'][270, :, :].data
        # End: mock code

        model_data = []
        for fac in factors:
            factor = self.cfg.get_factor_info(fac)
            if factor['env'] == 'mock':
                data = mock_data
            else:
                with Datacube(app='confidence_layer', env=factor['env']) as dc:
                    # Get the tile spec
                    gwf = GridWorkflow(dc.index, self.grid_spec)
                    indexed_tiles = gwf.list_cells(cell_index, product=factor['product'])
                    # load the data of the tile
                    dataset = gwf.load(tile=indexed_tiles[cell_index], measurements=factor['band'])
                    if 'time' in dataset.dims.keys():
                        data = dataset.data_vars[factor['band']][1, :, :].data
                    else:
                        data = dataset.data_vars[factor['band']][:, :].data
            if factor['name'].startswith('phat'): data[data < 0] = 0.0
            if factor['name'].startswith('mrvbf'): data[data > 10] = 10
            if factor['name'].startswith('modis'): data[data > 100] = 100
            model_data.append(data.ravel())
            del data
        del mock_data
        logging.info('loaded all factors for tile {}'.format(cell_index))
        return np.column_stack(model_data)

    def compute_confidence(self, cell_index):
        model = self.cfg.get_confidence_model()
        X = self.load_tile_data(cell_index, model.factors)
        P = model.predict_proba(X)[:, 1]
        del X
        return P.reshape(self.grid_spec.tile_resolution)

    def compute_confidence_filtered(self, cell_index):
        con_layer = self.compute_confidence(cell_index)
        env = self.cfg.get_env_of_product('wofs_summary')

        # Start: Mock code
        with Datacube(app='confidence_layer', env='prod') as dc:
            # Get the tile spec
            gwf = GridWorkflow(dc.index, self.grid_spec)
            indexed_tiles = gwf.list_cells(cell_index, product='ls5_nbar_albers')
            # load the data of the tile
            dataset = gwf.load(tile=indexed_tiles[cell_index], measurements=['blue'])
            data = dataset.data_vars['blue'][270, :, :].data.astype(DEFAULT_TYPE)
        # End: mock code

        # with Datacube(app='wofs_summary', env=env) as dc:
        #     gwf = GridWorkflow(dc.index, self.grid_spec)
        #     indexed_tile = gwf.list_cells(cell_index, product='wofs_summary')
        #     # load the data of the tile
        #     dataset = gwf.load(tile=indexed_tile[cell_index], measurements='frequency')
        #     data = dataset.data_vars['frequency'][1, :, :].data

        con_filtering = self.cfg.cfg.get('confidence_filtering')
        threshold = None
        if con_filtering:
            threshold = con_filtering.get('threshold')

        if threshold:
            data[con_layer <= threshold] = np.nan
        else:
            data[con_layer <= 0.10] = np.nan
        return data

    def compute_filename(self, cell_index):
        return Path(self.cfg.cfg['wofs_filtered_summary']['filename'].format(cell_index[0], cell_index[1]))

    def compute_and_write(self, cell_index):
        geo_box = self.grid_spec.tile_geobox(cell_index)

        # Compute dataset coords
        coords = dict()
        for dim in geo_box.dimensions:
            coords[dim] = Coordinate(netcdf_writer.netcdfy_coord(geo_box.coordinates[dim].values),
                                     geo_box.coordinates[dim].units)

        # Compute dataset variables
        var = Variable(dtype=np.dtype(DEFAULT_TYPE), nodata=DEFAULT_NODATA,
                       dims=geo_box.dimensions, units=geo_box.crs.units)
        vars = {self.cfg.cfg['wofs_filtered_summary']['confidence']: var,
                self.cfg.cfg['wofs_filtered_summary']['confidence_filtered']: var}
        vars_params = {self.cfg.cfg['wofs_filtered_summary']['confidence']: {},
                       self.cfg.cfg['wofs_filtered_summary']['confidence_filtered']: {}}

        # Get crs string
        crs = self.cfg.cfg['storage']['crs'] if self.cfg.cfg['storage'].get('crs') else DEFAULT_CRS

        # Create a dataset container
        netcdf_unit = create_netcdf_storage_unit(filename=self.compute_filename(cell_index),
                                                 crs=CRS(crs),
                                                 coordinates=coords,
                                                 variables=vars,
                                                 variable_params=vars_params)

        # Confidence layer: Fill variable data and set attributes
        netcdf_unit['confidence'][:] = netcdf_writer.netcdfy_data(self.compute_confidence(cell_index))
        # ToDo: check valid range and standard names
        netcdf_unit['confidence'].valid_range = [-25.0, 25.0]
        netcdf_unit['confidence'].standard_name = 'confidence layer'
        netcdf_unit['confidence'].coverage_content_type = 'modelResult'
        netcdf_unit['confidence'].long_name = 'Wofs Confidence Layer predicted by ??'

        # Confidence filtered wofs-stats frequency layer: Fill variable data and set attributes
        netcdf_unit['confidence_filtered'][:] = netcdf_writer.netcdfy_data(self.compute_confidence_filtered(cell_index))
        # ToDo: check valid range and standard names
        netcdf_unit['confidence_filtered'].valid_range = [-25.0, 25.0]
        netcdf_unit['confidence_filtered'].standard_name = 'confidence filtered layer'
        netcdf_unit['confidence_filtered'].coverage_content_type = 'modelResult'
        netcdf_unit['confidence_filtered'].long_name = 'Wofs-stats frequency confidence filtered layer??'

        # ToDo: Add global attributes

        # close the dataset
        netcdf_unit.close()