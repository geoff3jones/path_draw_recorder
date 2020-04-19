# in progress


# todo

- enable config of BSpline group on the CLI

# done

- light refactor of the file io into a separate namespace so it is not clogging the main function
- merge recorded files with continuation paths
- rename seedfile to settings file
- option to pick up where you left off if a settings file is found matching 
  - save initial mt state to settings file also containing current path and iter
  - optionally load current path and iter if detecting matching settings file
  - added CLI options to skip to a specific path and iteration
- optimise bspline drawing - sample at a lower level and draw as a line
- de-bug some of the bspline drawing where all the paths gather in the top left hand corner
- modularise the main thing
- make stroke recorder use the bspline character generation
- modularise elipse things
- modularise bspline
- add other template stroke shapes
  - bspline
- move the all random state stuff into once class
- save to disk pandas
- basic impl
- generate random ellipse arcs
- colour the ends of the arc
- record mouse path while clicked
- 