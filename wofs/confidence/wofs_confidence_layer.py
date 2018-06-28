import yaml
import pickle
from datacube.api import GridWorkflow
from datacube.index import Index
from datacube import Datacube

class FactorSpec(object):
    def __init__(self, name, env, product, band):
        self.name = name
        self.env = env
        self.product = product
        self.band = band

    def __str__(self):
        return "FactorSpec: %s at datacube %s product %s band %s" % \
               (self.name, self.env, self.product, self.band)

    def __repr__(self):
        return self.__str__()


class TileConfig(object):
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


class TileIO(object):
    def __init__(self, cfg):
        """

        :param TileConfig cfg:
        """
        self.cfg = cfg

    @staticmethod
    def get_tile(tile_spec, factor_spec):
        """
        Returns a dataset corresponding to tile_spec
        """

        with Datacube(app='confidence_layer', env=factor_spec['env']) as dc:
            gwf = GridWorkflow(dc.index)
            return gwf.load(tile=tile_spec, measurements=factor_spec['band'])

    def get_tiles_with_factor_spec(self, tile_spec, factors):
        for factor in factors:
            yield (factor, self.get_tile(tile_spec, factor))

    def store_tile_metadata(self, tile_spec):
        pass

    def store_tile(self):
        pass


class ProcessTile(object):
    def __init__(self, config, tile):
        self.cfg = config
        self.tile = tile

    def compute_tile(self):
        pass

    def run(self):
        pass