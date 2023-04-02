#!/bin/bash
@echo "Setuping kaggle API"
find . -name "kaggle\ \(*\).json" -exec mv {} ~/.kaggle/kaggle.json \;