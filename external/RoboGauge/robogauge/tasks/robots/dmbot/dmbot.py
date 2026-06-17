# -*- coding: utf-8 -*-
'''
@File    : dmbot.py
@Desc    : DMBot robot wrapper for RoboGauge
'''
from robogauge.tasks.robots.go2.go2 import Go2
from robogauge.tasks.robots.dmbot.dmbot_config import DMBotConfig


class DMBot(Go2):
    """DMBot uses the same RobotLab policy observation/action layout as Go2."""

    def __init__(self, cfg: DMBotConfig):
        super().__init__(cfg)
