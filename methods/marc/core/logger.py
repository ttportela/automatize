# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Lucas May Petry
'''
import sys
from datetime import datetime

class Logger(object):

    LOG_LINE = None
    INFO        = '[    INFO    ]'
    WARNING     = '[  WARNING   ]'
    ERROR       = '[   ERROR    ]'
    CONFIG      = '[   CONFIG   ]'
    RUNNING     = '[  RUNNING   ]'
    QUESTION    = '[  QUESTION  ]'

    def log(self, type, message):
        if Logger.LOG_LINE:
            sys.stdout.write("\n")
            sys.stdout.flush()
            Logger.LOG_LINE = None

        sys.stdout.write(str(type) + " " + self.cur_date_time() + " :: " + message + "\n")
        sys.stdout.flush()

    def log_dyn(self, type, message):
        line = str(type) + " " + self.cur_date_time() + " :: " + message
        sys.stdout.write("\r\x1b[K" + line.__str__())
        sys.stdout.flush()
        Logger.LOG_LINE = line

    def get_answer(self, message):
        return input(Logger.QUESTION + " " + self.cur_date_time() + " :: " + message)

    def cur_date_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")