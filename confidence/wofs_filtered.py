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


class Config(object):
    """
    Specify configuration parameters for wofs filtered summary product.
    """
    def __init__(self, config_file, metadata_template_file=None):
        with open(config_file, 'r') as cfg_stream:
            self.cfg = yaml.load(cfg_stream)

    def get_env_of_product(self, product_name):
        """
        Return datacube environment of the given product_name.
        """
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
                        (self.cfg['storage']['tile_size']['y'], self.cfg['storage']['tile_size']['x']),
                        (self.cfg['storage']['resolution']['y'], self.cfg['storage']['resolution']['x']))

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
    """
    Computes, writes of wofs filtered summary product for a single tile.
    """

    def __init__(self, config, grid_spec, tile_index):
        """
        Implements confidence filtering of wofs-summary frequency band and creation of
        filtered summary datasets.
        :param Config config: confidence Config object
        :param GridSpec grid_spec: A GridSpec object size 100kmx100km
        :param tile_index: Tuple like (17 , -39)
        """
        self.cfg = config
        self.grid_spec = grid_spec
        self.confidence_model = config.get_confidence_model()
        self.tile_index = tile_index
        self.factor_sources = self._get_factor_datasets()

    def _get_factor_datasets(self):
        dts = []
        for fac in self.confidence_model.factors:
            factor = self.cfg.get_factor_info(fac)
            with Datacube(app='confidence_layer', env=factor['env']) as dc:
                gwf = GridWorkflow(dc.index, self.grid_spec)
                obs = gwf.cell_observations(cell_index=self.tile_index, product=factor['product'])
                for ds in obs[self.tile_index]['datasets']:
                    dts.append(ds)
        return dts

    def load_tile_data(self, factors):
        """
        Load and return factor data for confidence band prediction.
        :param factors: List of factor info as given by Config
        """

        model_data = []
        for fac in factors:
            factor = self.cfg.get_factor_info(fac)
            with Datacube(app='confidence_layer', env=factor['env']) as dc:
                gwf = GridWorkflow(dc.index, self.grid_spec)
                indexed_tiles = gwf.list_cells(self.tile_index, product=factor['product'])
                # load the data of the tile
                dataset = gwf.load(tile=indexed_tiles[self.tile_index], measurements=[factor['band']])
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
        logging.info('loaded all factors for tile {}'.format(self.tile_index))
        return np.column_stack(model_data)

    def compute_confidence(self):
        """
        Return the confidence band predicted by the model.factors. The pre-trained model
        is loaded from the path specified by the config.
        """

        model = self.cfg.get_confidence_model()
        X = self.load_tile_data(model.factors)
        P = model.predict_proba(X)[:, 1]
        del X
        return P.reshape(self.grid_spec.tile_resolution)

    def compute_confidence_filtered(self):
        """
        Return the wofs filtered summary band data that is 10% filtered by confidence band.
        """

        con_layer = self.compute_confidence()
        env = self.cfg.get_env_of_product('wofs_summary')

        with Datacube(app='wofs_summary', env=env) as dc:
            gwf = GridWorkflow(dc.index, self.grid_spec)
            indexed_tile = gwf.list_cells(self.tile_index, product='wofs_summary')
            # load the data of the tile
            dataset = gwf.load(tile=indexed_tile[self.tile_index], measurements=['frequency'])
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
        """
        Return a Path object of wofs filtered summary NetCDF file corresponding to the current tile.
        """

        file_name = self.cfg.cfg['wofs_filtered_summary']['filename'].format(self.tile_index[0],
                                                                             self.tile_index[1])
        return Path(self.cfg.cfg['wofs_filtered_summary']['filtered_summary_dir']) / Path(file_name)

    def compute_and_write(self):
        """
        Computes the wofs confidence and filtered summary bands and write to the
        corresponding NetCDF file. The file template and location etc are read from the configs.
        """

        geo_box = self.grid_spec.tile_geobox(self.tile_index)

        # Compute metadata
        env = self.cfg.get_env_of_product('wofs_filtered_summary')
        with Datacube(app='wofs-confidence', env=env) as dc:
            product = dc.index.products.get_by_name('wofs_filtered_summary')
        extent = self.grid_spec.tile_geobox(self.tile_index).extent
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

        band1 = self.cfg.cfg['wofs_filtered_summary']['confidence']
        band2 = self.cfg.cfg['wofs_filtered_summary']['confidence_filtered']

        vars = {band1: spatial_var,
                band2: spatial_var}
        vars_params = {band1: {},
                       band2: {}}
        global_atts = self.cfg.cfg['global_attributes']

        # Get crs string
        crs = self.cfg.cfg['storage']['crs'] if self.cfg.cfg['storage'].get('crs') else DEFAULT_CRS

        # Create a dataset container
        netcdf_unit = create_netcdf_storage_unit(filename=self.get_filtered_uri(),
                                                 crs=CRS(crs),
                                                 coordinates=coords,
                                                 variables=vars,
                                                 global_attributes=global_atts,
                                                 variable_params=vars_params)

        # Confidence layer: Fill variable data and set attributes
        confidence = self.compute_confidence()
        netcdf_unit[band1][:] = netcdf_writer.netcdfy_data(confidence)
        netcdf_unit[band1].units = '1'
        netcdf_unit[band1].valid_range = [0, 1.0]
        netcdf_unit[band1].coverage_content_type = 'modelResult'
        netcdf_unit[band1].long_name = \
            'Wofs Confidence Layer predicted by {}'.format(self.confidence_model.factors.__str__())

        # Confidence filtered wofs-stats frequency layer: Fill variable data and set attributes
        confidence_filtered = self.compute_confidence_filtered()
        netcdf_unit[band2][:] = netcdf_writer.netcdfy_data(confidence_filtered)
        netcdf_unit[band2].units = '1'
        netcdf_unit[band2].valid_range = [0, 1.0]
        netcdf_unit[band2].coverage_content_type = 'modelResult'
        netcdf_unit[band2].long_name = 'WOfS-Stats frequency confidence filtered layer'

        # Metadata
        dataset_data = DataArray(data=[metadata], dims=('time',))
        netcdf_writer.create_variable(netcdf_unit, 'dataset', dataset_data, zlib=True)
        netcdf_unit['dataset'][:] = netcdf_writer.netcdfy_data(dataset_data.values)

        netcdf_unit.close()


class TileProcessor(object):
    """
    Identify tiles to br processed.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_tile_list(self):
        """
        Return the list of tiles for Australian continental run.
        Each tile is a tuple, eg (17, -39)
        """
        list_path = self.cfg.cfg['coverage']['tile_list_path']
        tiles = []
        with open(list_path, 'r') as f:
            for line in f:
                num = [int(str_num) for str_num in line[:-1].split()]
                tiles.append((num[0], num[1]))
        return tiles

    def get_tiles_to_process(self):
        """
        Identify the tiles not present in the filtered summary directory specified bt the configs
        for a australian continental coverage.
        """

        def file_exists(tile):
            file_name = self.cfg.cfg['wofs_filtered_summary']['filename'].format(tile[0], tile[1])
            file_path = Path(self.cfg.cfg['wofs_filtered_summary']['filtered_summary_dir']) / Path(file_name)
            return file_path.is_file()

        tile_list = self.get_tile_list()
        # remove existing tiles from the list
        return [tile for tile in tile_list if not file_exists(tile)]

    def print_tiles_to_process(self):
        """
        Print the tiles yet to be processed on to the screen
        """
        tiles = self.get_tiles_to_process()
        for tile in tiles:
            print('{} {}'.format(tile[0], tile[1]))


def print_tiles(ctx, param, value):
    if value:
        processor = TileProcessor(Config(ctx.params['config']))
        processor.print_tiles_to_process()
        ctx.exit()


@click.command()
@click.option('--config', help='The config file')
@click.option('--retile', is_flag=True, callback=print_tiles,
              help='Identify tiles to be computed for a new run')
@click.option('--tile', nargs=2, type=click.Tuple([int, int]), help='The tile index')
def main(config, retile, tile):
    if config:
        cfg = Config(config)
    else:
        cfg = Config('/g/data/u46/users/aj9439/wofs/configs/template_client.yaml')
    if tile:
        grid_spec = cfg.get_grid_spec()
        wf = WofsFiltered(cfg, grid_spec, tile)
        wf.compute_and_write()


if __name__ == '__main__':
    main()
