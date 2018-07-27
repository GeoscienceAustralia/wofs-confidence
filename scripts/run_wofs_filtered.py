#!/usr/bin/env python3

from pathlib import Path
import pydash
import subprocess
from confidence import Config


class TileProcessor(object):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_tile_list(self):
        list_path = self.cfg.cfg['coverage']['tile_list_path']
        tiles = []
        with open(list_path, 'r') as f:
            for line in f:
                num = [int(str_num) for str_num in line[:-1].split()]
                tiles.append((num[0], num[1]))
        return tiles

    def get_tiles_to_process(self):

        def file_exists(tile):
            file_name = self.cfg.cfg['wofs_filtered_summary']['filename'].format(tile[0], tile[1])
            file_path = Path(self.cfg.cfg['wofs_filtered_summary']['filtered_summary_dir']) / Path(file_name)
            return file_path.is_file()

        tile_list = self.get_tile_list()
        # remove existing tiles from the list
        return [tile for tile in tile_list if not file_exists(tile)]

    @staticmethod
    def get_number_of_nodes():
        # ToDo
        return 4

    def run(self):
        tiles = self.get_tiles_to_process()
        nodes = self.get_number_of_nodes()
        tile_lists = pydash.arrays.chunk(tiles, nodes)
        for tile_list in tile_lists:
            # process tile_list on a single node
            cmd_template = 'python ../confidence/wofs_filtered.py --config {} --cell {} {}'
            for tile in tile_list:
                cmd = cmd_template.format('../configs/template_client.yaml',
                                          tile[0], tile[1])
                print(cmd)
                subprocess.run(cmd, check=True, shell=True,
                               cwd='/home/547/aj9439/PycharmProjects/wofs-confidence/tests')


if __name__ == '__main__':
    config = Config('/home/547/aj9439/PycharmProjects/wofs-confidence/configs/template_client.yaml')
    processor = TileProcessor(config)
    processor.run()
