#!/bin/bash
find . -name "kaggle.json" -exec mv {} ~/.kaggle/kaggle.json \;
chmod 600 ~/.kaggle/kaggle.json
