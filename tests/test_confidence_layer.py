from wofs.confidence import Config, WofsFiltered
from datacube import Datacube
from datacube.model import GridSpec
from datacube.utils.geometry import CRS


# Test Config class
def test_configs():
    cfg = Config('../configs/template_client.yaml')
    assert cfg.cfg is not None
    assert cfg.get_confidence_model() is not None
    assert cfg.get_env_of_product('wofs_summary_filtered') == 'prod'
    assert cfg.get_factor_info('mrvbf')['env'] == 'mock'
    assert cfg.get_factor_info('mrvbf')['band'] == 'band1'


# Test WofsFiltered class
def test_wofs_filtered():
    cfg = Config('../configs/template_client.yaml')
    dc = Datacube(app='test-wofs', env='prod')
    # grid_spec = dc.index.products.get_by_name('ls5_nbar_albers').grid_spec
    grid_spec = GridSpec(crs=CRS('EPSG:3577'), tile_size=(10000, 10000), resolution=(-25, 25))
    # import ipdb; ipdb.set_trace()
    cell_index = (-120, -120)
    wf = WofsFiltered(cfg, grid_spec)
    confidence = wf.compute_confidence(cell_index)
    filtered = wf.compute_confidence_filtered(cell_index)
    wf.compute_and_write(cell_index)


if __name__ == "__main__":
    test_configs()
    test_wofs_filtered()
