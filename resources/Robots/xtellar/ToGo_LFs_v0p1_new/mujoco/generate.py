#!/usr/bin/env python

import argparse
from pathlib import Path
from marionette_emgen import em_generate
from marionette_emgen.csv_config import Config
from marionette_emgen.mjcf import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o',
                        help='Output path')
    parser.add_argument('--rgba', default='#C0C0C0FF',
                        help='Main color in Hex format')
    parser.add_argument('--jpos-noise', type=float, default=0.,
                        help='Joint position sensor noise')
    parser.add_argument('--jvel-noise', type=float, default=0.,
                        help='Joint velocity sensor noise')
    parser.add_argument('--jtor-noise', type=float, default=0.,
                        help='Joint torque sensor noise')
    parser.add_argument('--gyro-noise', type=float, default=0.,
                        help='Gyroscope sensor noise')
    parser.add_argument('--accl-noise', type=float, default=0.,
                        help='Accelerometer sensor noise')
    parser.add_argument('--orie-noise', type=float, default=0.,
                        help='IMU orientation estimation noise')

    args = parser.parse_args()

    cfg = Config.load('../csv')

    em_dict = dict(
        cfg=cfg,
        taghead_body=lambda **kwargs: em_taghead_body(cfg, **kwargs),
        tag_joint=lambda **kwargs: em_tag_joint(cfg, **kwargs),
        tag_inertial=lambda **kwargs: em_tag_inertial(cfg, **kwargs),
        tag_motor=lambda **kwargs: em_tag_motor(cfg, **kwargs),
        tag_site=lambda **kwargs: em_tag_site(cfg, **kwargs),
        _main_rgba=(
            int(args.rgba[1:3], 16)/255.,
            int(args.rgba[3:5], 16)/255.,
            int(args.rgba[5:7], 16)/255.,
            int(args.rgba[7:9], 16)/255.,
        ),
        _jpos_noise=f'{args.jpos_noise:.3f}',
        _jvel_noise=f'{args.jvel_noise:.3f}',
        _jtor_noise=f'{args.jtor_noise:.3f}',
        _gyro_noise=f'{args.gyro_noise:.3f}',
        _accl_noise=f'{args.accl_noise:.3f}',
        _orie_noise=f'{args.orie_noise:.3f}',
    )

    em_generate(
        Path('ToGo_LFs_v0p1_prototype.xml.em'),
        args.output,
        em_dict=em_dict,
    )
