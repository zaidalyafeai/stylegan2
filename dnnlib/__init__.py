# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from . import submission

from .submission.run_context import RunContext

from .submission.submit import SubmitTarget
from .submission.submit import PathType
from .submission.submit import SubmitConfig
from .submission.submit import submit_run
from .submission.submit import submit_diagnostic
from .submission.submit import get_path_from_template
from .submission.submit import convert_path
from .submission.submit import make_run_dir_path

from .util import EasyDict

submit_config: SubmitConfig = None # Package level variable for SubmitConfig which is only valid when inside the run function.
