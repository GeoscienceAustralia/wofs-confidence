#!/usr/bin/env python3

import pydash
import subprocess
import time


class TileProcessor(object):
    def __init__(self, cfg_yaml):
        pass

    def get_tile_list(self):
        # ToDo
        return [(17, -39), (17, -38)]

    def get_tiles_to_process(self):
        # ToDo
        return [(17, -38), (17, -39), (17, -40), (16, -39)]

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
    processor = TileProcessor('stuff')
    processor.run()
