#!/bin/bash

source /etc/network_turbo

# 获取远程最新代码
git pull

# 强制切换到远程分支（会进入 detached HEAD 状态）
git checkout origin/2025.11.5 -f
