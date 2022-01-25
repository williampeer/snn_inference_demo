#!/usr/bin/env bash

mkdir archive
cp -r figures archive/
cp -r Logs archive/
cp -r saved archive/
cp -r Test/figures archive/
cp -r Test/Logs archive/
cp -r Test/saved archive/

zip -r archive.zip archive/

