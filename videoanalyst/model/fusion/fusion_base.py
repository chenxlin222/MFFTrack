# -*- coding: utf-8 -*
from videoanalyst.utils import Registry

TRACK_FUSIONS = Registry('TRACK_FUSIONS')
VOS_FUSIONS = Registry('VOS_FUSIONS')

TASK_FUSIONS = dict(
    track=TRACK_FUSIONS,
    vos=VOS_FUSIONS,
)
