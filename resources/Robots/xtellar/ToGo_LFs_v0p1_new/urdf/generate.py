#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

from marionette_emgen import em_generate
from marionette_emgen.csv_config import Config
from marionette_emgen.urdf import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o',
                        help='Output path')
    parser.add_argument('--no-virtual-links', action='store_true',
                        help='Do not generate with virtual links')
    parser.add_argument('--mujoco', action='store_true',
                        help='Generate MuJoCo related tags to validate URDF')
    parser.add_argument('--rgba', default='#C0C0C0FF',
                        help='Main color in Hex format')
    parser.add_argument('--mesh-prefix', default='../meshes/',
                        help='Prefix prepended to the mesh paths')

    args = parser.parse_args()

    cfg = Config.load('../csv')

    def _tag_link(**kwargs):
        mesh_path = kwargs.get('mesh_path')
        visual_mesh_path = f'visual/{mesh_path}'
        collision_mesh_path = f'collision/{mesh_path}'
        kwargs['mesh_path'] = visual_mesh_path
        kwargs['collision_mesh_path'] = collision_mesh_path
        return em_tag_link(cfg, mesh_scale=(0.001, 0.001, 0.001), mesh_prefix=args.mesh_prefix, **kwargs)

    em_dict = dict(
        cfg=cfg,
        tag_link=_tag_link,
        tag_joint=lambda **kwargs: em_tag_joint(cfg, **kwargs),
        _no_virtual_links=args.no_virtual_links,
        _mujoco=args.mujoco,
        _main_rgba=(
            int(args.rgba[1:3], 16)/255.,
            int(args.rgba[3:5], 16)/255.,
            int(args.rgba[5:7], 16)/255.,
            int(args.rgba[7:9], 16)/255.,
        ),
    )

    if args.output is None:
        if args.no_virtual_links:
            args.output = 'ToGo_LFs_v0p1_prototype_novlnk.urdf'

    em_generate(
        Path('ToGo_LFs_v0p1_prototype.urdf.em'),
        args.output,
        em_dict=em_dict,
    )
