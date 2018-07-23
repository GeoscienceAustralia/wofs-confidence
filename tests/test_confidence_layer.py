from confidence import Config, WofsFiltered
from datacube import Datacube
from datacube.api import GridWorkflow
from datacube.model import GridSpec
from datacube.utils.geometry import CRS
import matplotlib.pyplot as plt
import rasterio


# Test Config class
def test_configs():
    cfg = Config('../configs/template_client.yaml')
    assert cfg.cfg is not None
    assert cfg.get_confidence_model() is not None
    assert cfg.get_env_of_product('wofs_summary_filtered') == 'dev'
    assert cfg.get_factor_info('mrvbf')['env'] == 'dev'
    assert cfg.get_factor_info('mrvbf')['band'] == 'band1'


# Test WofsFiltered class
def test_wofs_filtered():
    cfg = Config('../configs/template_client.yaml')
    grid_spec = GridSpec(crs=CRS('EPSG:3577'), tile_size=(100000, 100000), resolution=(-25, 25))
    cell_index = (17, -39)
    wf = WofsFiltered(cfg, grid_spec, cell_index)
    confidence = wf.compute_confidence(cell_index)
    filtered = wf.compute_confidence_filtered()

    # Display images: to be removed later
    with Datacube(app='wofs_summary', env='dev') as dc:
        gwf = GridWorkflow(dc.index, grid_spec)
        indexed_tile = gwf.list_cells(cell_index, product='wofs_statistical_summary')
        # load the data of the tile
        dataset = gwf.load(tile=indexed_tile[cell_index], measurements=['frequency'])
        frequency = dataset.data_vars['frequency'].data.ravel().reshape(grid_spec.tile_resolution)

    # Check with previous run
    with rasterio.open('confidenceFilteredWOfS_17_-39_epsilon=10.tiff') as f:
        data = f.read(1)
    plt.subplot(221)
    plt.imshow(frequency)
    plt.subplot(222)
    plt.imshow(data)
    plt.subplot(223)
    plt.imshow(confidence)
    plt.subplot(224)
    plt.imshow(filtered)
    plt.show()
    wf.compute_and_write()


if __name__ == "__main__":
    test_configs()
    test_wofs_filtered()
