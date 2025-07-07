# Copyright (c) 2025, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch


def small_init_init_method(dim, use_mup_outer=False, mup_init_scale=1.0):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution."""
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor, use_mup=use_mup_outer):
        # if use_mup:
        #     mup.init.normal_(tensor, mean=0.0, std=std)
        #     with torch.no_grad():
        #         tensor.mul_(mup_init_scale)
        #     return tensor
        # else:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim, use_mup_outer=False, mup_init_scale=1.0):
    std = 2 / n_layers / math.sqrt(dim)

    def init_(tensor, use_mup=use_mup_outer):
        # if use_mup:
        #     mup.init.normal_(tensor, mean=0.0, std=std)
        #     with torch.no_grad():
        #         tensor.mul_(mup_init_scale)
        #     return tensor
        # else:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

