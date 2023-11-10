#!/bin/bash

ETH_USERNAME=kpius
PROJECT_NAME=si_bees
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}/log/

scp -r ${ETH_USERNAME}@tik42x.ethz.ch:${DIRECTORY}/$1 .