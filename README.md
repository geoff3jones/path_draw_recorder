# About 

This is a fairly basic tool for recording mouse paths drawn by the user. It can
generate guide paths and display them for a user to trace/copy

# Installation

This was developed on python 3.8.2 but may work on other recent python versions.
Install the requirements and away you go. You only need pyarrow if you intend to
save with the Apache parquet format.

# Usage

- entry point is `python main.py [options]`
- CLI is configurable see main.py -h for mor information
- use space-bar to advance to the next guide path
- click and drag the mouse pointer along the white path from the green end to the red
  - if you screw up you can re-click to draw again, only the last path is saved
- if you mangle your output path it will try to save the output to the pwd