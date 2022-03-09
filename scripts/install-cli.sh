#!/bin/bash

BASE_STRING='export PATH=$PATH:'
if [[ -f "${HOME}/.zshrc" ]] ; then
    echo "$BASE_STRING$PWD" >> ${HOME}/.zshrc
    source ${HOME}/.zshrc
    echo "[INFO] installed for zsh"
fi

if [[ -f "${HOME}/.bashrc" ]] ; then
    echo "$BASE_STRING$PWD" >> ${HOME}/.bashrc
    source ${HOME}/.bashrc
    echo "[INFO] installed for bash"
fi