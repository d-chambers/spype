# -*- coding: utf-8 -*-
import spype.callbacks
# import spype.pype
import spype.utils
from spype.core.task import Task, task, pype_input, forward
from spype.core.wrap import Wrap
from spype.core.pype import Pype
from spype.utils import context
from spype.version import __version__
from spype.exceptions import ExitTask

from spype.constants import PYPE_FIXTURES, TASK_FIXTURES

# set additional aliases
options = context
set_options = context
