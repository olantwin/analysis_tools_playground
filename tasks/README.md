# FairTask approach

## Example use:

`python task_runner.py -g geofile_full.Ntuple-TGeant4.root -f numu_dig_PR.root track vertex`

## TODO

* Pattern matching not yet done (use atomic script for now)
* I/O very buggy, not sure how to register STL branches successfully (issues with dictionarys which doesn't occur with atomic scripts)

## Limitations/challenges

* TaskName not properly set when inheriting in python
* Problems with STL containers?
* Have to be diligent in updating BranchList and TaskList, but could use this for dependency checking of tasks!
