import yaml
import pickle
from datacube.api import GridWorkflow
from datacube import Datacube
import numpy as np
import logging


class Config(object):
    def __init__(self, config_file, metadata_template_file):
        with open(config_file, 'r') as cfg_stream:
            self.cfg = yaml.load(cfg_stream)
        with open(metadata_template_file, 'r') as template_stream:
            self.metadata_template = yaml.load(template_stream)

    def get_env_of_product(self, product_name):
        for dc in self.cfg['datacubes']:
            for dc_env in dc.keys():
                if product_name in dc[dc_env]:
                    return dc_env

    def get_factor_info(self, factor_name):
        for factor in self.cfg['factors']:
            if factor_name == factor['name']:
                if factor.get('flag'):
                    return {'env': self.get_env_of_product(factor['product']),
                            'product': factor['product'],
                            'band': factor['band'],
                            'flag': factor['flag']}
                else:
                    return {'env': self.get_env_of_product(factor['product']),
                            'product': factor['product'],
                            'band': factor['band']}

    def get_confidence_model(self):
        """
            The model representation we use are sklearn.LogisticRegression objects with
            added 'factor' specifications, i.e. if model is a sklearn linear_model object,
            model.factors will have factor specifications. A factor is a FactorSpec object
            that hold information which is required to access factor data, typically, from
            a datacube.
        """
        model_path = self.cfg['confidence_outputs']['trained_model_path']
        
        # read the trained model
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class WofsFiltered(object):
    def __init__(self, config, grid_spec):
        self.cfg = config
        self.grid_spec = grid_spec

    def load_data(self, cell_index, factors):
        model_data = []
        for factor in factors:
            with Datacube(app='confidence_layer', env=factor['env']) as dc:
                product = dc.index.products.get_by_name(factor['product'])
                gwf = GridWorkflow(dc.index, self.grid_spec)
                geobox = gwf.grid_spec.tile_geobox(cell_index)
                # Create the Tile object
                data = gwf.load(tile=self.grid_spec, measurements=factor['band'])
            # ToDo: Get the thresholds/bounds from config
            if factor['name'].startswith('phat'): data[data < 0] = 0.0
            if factor['name'].startswith('mrvbf'): data[data > 10] = 10
            if factor['name'].startswith('modis'): data[data > 100] = 100
            model_data.append(data.ravel())
            del data
        logging.info('loaded all factors for tile {}'.format(self.grid_spec))
        return np.column_stack(model_data)

    def compute_confidence(self):
        model = self.cfg.get_confidence_model()
        X = self.load_data(model.factors)
        P = model.predict_proba(X)[:, 1]
        del X
        # ToDo: find cell_shape
        return P.reshape(cell_shape)
