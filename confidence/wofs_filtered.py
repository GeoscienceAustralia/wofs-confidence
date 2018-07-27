#!/usr/bin/env python3

import yaml
import pickle
from datacube.api import GridWorkflow
from datacube.model import GridSpec, Variable
from datacube.model.utils import make_dataset
from datacube.utils.geometry import GeoBox, Coordinate, CRS
from datacube.utils import datetime_to_seconds_since_1970
from datacube.storage import netcdf_writer
from datacube.storage.storage import create_netcdf_storage_unit
from datacube import Datacube
import numpy as np
from xarray import DataArray
from datetime import datetime
from pathlib import Path
import logging
import click

try:
    from yaml import CSafeDumper as SafeDumper
except ImportError:
    from yaml import SafeDumper

DEFAULT_CRS = 'EPSG:3577'
DEFAULT_NODATA = np.nan
DEFAULT_FLOAT_NODATA = -1.0
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

    def get_grid_spec(self):
        return GridSpec(CRS(self.cfg['storage']['crs']),
                        (self.cfg['storage']['tile_size']['x'], self.cfg['storage']['tile_size']['y']),
                        (self.cfg['storage']['resolution']['x'], self.cfg['storage']['resolution']['y']))

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
    def __init__(self, config, grid_spec, cell_index):
        """
        Implements confidence filtering of wofs-summary frequency band and creation of
        filtered summary datasets.
        :param Config config:
        :param GridSpec grid_spec:
        :param cell_index:
        """
        self.cfg = config
        self.grid_spec = grid_spec
        self.confidence_model = config.get_confidence_model()
        self.cell_index = cell_index
        self.factor_sources = self._get_factor_datasets()

    def _get_factor_datasets(self):
        dts = []
        for fac in self.confidence_model.factors:
            factor = self.cfg.get_factor_info(fac)
            with Datacube(app='confidence_layer', env=factor['env']) as dc:
                gwf = GridWorkflow(dc.index, self.grid_spec)
                obs = gwf.cell_observations(cell_index=self.cell_index, product=factor['product'])
                for ds in obs[self.cell_index]['datasets']:
                    dts.append(ds)
        return dts

    def load_tile_data(self, cell_index, factors):
        model_data = []
        for fac in factors:
            factor = self.cfg.get_factor_info(fac)
            with Datacube(app='confidence_layer', env=factor['env']) as dc:
                gwf = GridWorkflow(dc.index, self.grid_spec)
                indexed_tiles = gwf.list_cells(cell_index, product=factor['product'])
                # load the data of the tile
                dataset = gwf.load(tile=indexed_tiles[cell_index], measurements=[factor['band']])
                data = dataset.data_vars[factor['band']].data

            # Rescale where needed: Keep an eye on this since this is to do with different scaling factors used during
            # training than what is on datacube
            if factor['name'].startswith('phat'): data = data * 100.0

            if factor['name'].startswith('phat'): data[data < 0.0] = 0.0
            if factor['name'].startswith('mrvbf'): data[data > 10] = 10
            if factor['name'].startswith('modis'): data[data > 100] = 100
            model_data.append(data.ravel())
            del data
        # del mock_data
        logging.info('loaded all factors for tile {}'.format(cell_index))
        return np.column_stack(model_data)

    def compute_confidence(self, cell_index):
        model = self.cfg.get_confidence_model()
        X = self.load_tile_data(cell_index, model.factors)
        P = model.predict_proba(X)[:, 1]
        del X
        return P.reshape(self.grid_spec.tile_resolution)

    def compute_confidence_filtered(self):
        con_layer = self.compute_confidence(self.cell_index)
        env = self.cfg.get_env_of_product('wofs_statistical_summary')

        with Datacube(app='wofs_summary', env=env) as dc:
            gwf = GridWorkflow(dc.index, self.grid_spec)
            indexed_tile = gwf.list_cells(self.cell_index, product='wofs_statistical_summary')
            # load the data of the tile
            dataset = gwf.load(tile=indexed_tile[self.cell_index], measurements=['frequency'])
            data = dataset.data_vars['frequency'].data.ravel().reshape(self.grid_spec.tile_resolution)

        con_filtering = self.cfg.cfg.get('confidence_filtering')
        threshold = None
        if con_filtering:
            threshold = con_filtering.get('threshold')

        if threshold:
            data[con_layer <= threshold] = DEFAULT_FLOAT_NODATA
        else:
            data[con_layer <= 0.10] = DEFAULT_FLOAT_NODATA

        return data

    def get_filtered_uri(self):
        file_name = self.cfg.cfg['wofs_filtered_summary']['filename'].format(self.cell_index[0],
                                                                             self.cell_index[1])
        return Path(self.cfg.cfg['wofs_filtered_summary']['filtered_summary_dir']) / Path(file_name)

    def compute_and_write(self):
        geo_box = self.grid_spec.tile_geobox(self.cell_index)

        # Compute metadata
        env = self.cfg.get_env_of_product('wofs_filtered_summary')
        with Datacube(app='wofs-confidence', env=env) as dc:
            product = dc.index.products.get_by_name('wofs_filtered_summary')
        extent = self.grid_spec.tile_geobox(self.cell_index).extent
        center_time = datetime.now()
        uri = self.get_filtered_uri()
        dts = make_dataset(product=product, sources=self.factor_sources,
                           extent=extent, center_time=center_time, uri=uri)
        metadata = yaml.dump(dts.metadata_doc, Dumper=SafeDumper, encoding='utf-8')

        # Compute dataset coords
        coords = dict()
        coords['time'] = Coordinate(netcdf_writer.netcdfy_coord(
            np.array([datetime_to_seconds_since_1970(center_time)])), ['seconds since 1970-01-01 00:00:00'])
        for dim in geo_box.dimensions:
            coords[dim] = Coordinate(netcdf_writer.netcdfy_coord(geo_box.coordinates[dim].values),
                                     geo_box.coordinates[dim].units)

        # Compute dataset variables
        spatial_var = Variable(dtype=np.dtype(DEFAULT_TYPE), nodata=DEFAULT_FLOAT_NODATA,
                               dims=('time',) + geo_box.dimensions,
                               units=('seconds since 1970-01-01 00:00:00',) + geo_box.crs.units)
        vars = {self.cfg.cfg['wofs_filtered_summary']['confidence']: spatial_var,
                self.cfg.cfg['wofs_filtered_summary']['confidence_filtered']: spatial_var}
        vars_params = {self.cfg.cfg['wofs_filtered_summary']['confidence']: {},
                       self.cfg.cfg['wofs_filtered_summary']['confidence_filtered']: {}}

        # Get crs string
        crs = self.cfg.cfg['storage']['crs'] if self.cfg.cfg['storage'].get('crs') else DEFAULT_CRS

        # Create a dataset container
        netcdf_unit = create_netcdf_storage_unit(filename=self.get_filtered_uri(),
                                                 crs=CRS(crs),
                                                 coordinates=coords,
                                                 variables=vars,
                                                 variable_params=vars_params)

        # Confidence layer: Fill variable data and set attributes
        confidence = self.compute_confidence(self.cell_index)
        netcdf_unit['confidence'][:] = netcdf_writer.netcdfy_data(confidence)
        netcdf_unit['confidence'].units = '1'
        netcdf_unit['confidence'].valid_range = [-1.0, 1.0]
        netcdf_unit['confidence'].standard_name = 'confidence'
        netcdf_unit['confidence'].coverage_content_type = 'modelResult'
        netcdf_unit['confidence'].long_name = \
            'Wofs Confidence Layer predicted by {}'.format(self.confidence_model.factors.__str__())

        # Confidence filtered wofs-stats frequency layer: Fill variable data and set attributes
        confidence_filtered = self.compute_confidence_filtered()
        netcdf_unit['confidence_filtered'][:] = netcdf_writer.netcdfy_data(confidence_filtered)
        netcdf_unit['confidence_filtered'].units = '1'
        netcdf_unit['confidence_filtered'].valid_range = [-1.0, 1.0]
        netcdf_unit['confidence_filtered'].standard_name = 'confidence_filtered'
        netcdf_unit['confidence_filtered'].coverage_content_type = 'modelResult'
        netcdf_unit['confidence_filtered'].long_name = 'Wofs-stats frequency confidence filtered layer'

        # Metadata
        dataset_data = DataArray(data=[metadata], dims=('time',))
        netcdf_writer.create_variable(netcdf_unit, 'dataset', dataset_data, zlib=True)
        netcdf_unit['dataset'][:] = netcdf_writer.netcdfy_data(dataset_data.values)

        netcdf_unit.close()


@click.command()
@click.option('--config', help='The config file')
@click.option('--cell', nargs=2, type=click.Tuple([int, int]), help='The cell index')
def main(config, cell):
    cfg = Config(config)
    grid_spec = cfg.get_grid_spec()
    wf = WofsFiltered(cfg, grid_spec, cell)
    wf.compute_and_write()


if __name__ == '__main__':
    main()