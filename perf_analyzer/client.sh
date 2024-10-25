#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
THIS_SCRIPT_DIR="$(realpath --relative-to="${PWD}" "$(dirname "$0")")"
INPUT_DATA="${THIS_SCRIPT_DIR}/input-data.json"

perf_analyzer -u 127.0.0.1:8001 \
  -i HTTP \
  -m BART \
  --measurement-mode count_windows \
  --measurement-request-count 100 \
  --input-data ${INPUT_DATA} \
  --concurrency-range 4:16:4 \
  -v
