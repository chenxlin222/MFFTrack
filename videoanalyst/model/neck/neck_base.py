# -*- coding: utf-8 -*
from videoanalyst.utils import Registry

TRACK_NECKS = Registry('TRACK_NECKS')
VOS_NECKS = Registry('VOS_NECKS')
TASK_NECKS = dict(
    track=TRACK_NECKS,
    vos=VOS_NECKS,
)