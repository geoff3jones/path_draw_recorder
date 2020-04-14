import re
import sys
import subprocess
import os

if sys.version_info[0] < 3:
    print("OOPS: This script requires python version 3 try running it with a newer version.")
    sys.exit()
if sys.version_info[1] < 7:
    print(
    """WARNING: This script was only tested using python 3.8.x and may function
         incorrectly on earlier python versions.""")
    # sys.exit()

msg = \
"""This will run the capture app and then save a file in this directory.
Geoff would really appreciate it if you were to then share it with him :)
although do be warned that it may take up to half an hour to complete!"""
print(msg)
response = input("Do you wish to continue? [y/N]")
if re.match(r'[yY][eE][sS]|[yY]', response) is None:
    print("no worries, maybe later then?")
    sys.exit()

print("Awesome!")
msg = \
"""...
Right ok here's how it works:
    - Press space to advance thorugh the paths.
    - Use the left mouse button to drag and trace over the paths.
    - You can re draw a path with the mouse as many times as you like and it
      will only save the most recent one.
    - Press Esc to quit - your progress until then will be saved but you will
      have to start form the begginning next time.
    - Clicking the X in the top right hand corner does nothing, use Esc to quit.
"""
print(msg)
input('continue...')

root_path = os.path.dirname( os.path.realpath(__file__))
main_file = os.path.join(root_path, 'src', 'main.py')
out_file  = os.path.join(root_path, 'standard_set')
command = [sys.executable, main_file, "--seed", "1000000", "--outfile",
           out_file, "--outtype", "parquet", "-i", "5", "-n", "50"]
print("running:\n",' '.join(command))
subprocess.run(command)
print("Gosh Thanks!!")
