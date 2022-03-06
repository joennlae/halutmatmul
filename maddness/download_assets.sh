#!/bin/bash

rm -rf assets
mkdir -p assets
cd assets
curl -L -o repo.tar.gz https://github.com/joennlae/bolt/archive/7902ce4c698f7464a515036bf552c6fafb95a5e9.tar.gz
tar -xvf repo.tar.gz --strip-components=3 bolt-7902ce4c698f7464a515036bf552c6fafb95a5e9/experiments/assets