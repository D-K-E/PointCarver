###############################
Quick Start with Point Carver
###############################

Welcome to point carver. Point carver has a single job: carve a way from
selected point to the chosen direction.

If you have segmentation jobs that require manual work, 
it could help you semi automate the process.

Requirements
==============

The core functionality depends on :code:`scipy` and :code:`numpy`. The
interface requires :code:`pyside2` and :code:`pillow`. 

If you are using conda for managing your packages and environments
all of those requirements can be installed with following:

- :code:`git clone https://github.com/D-K-E/PointCarver.git`

- :code:`cd PointCarver`

- :code:`conda create --name pointcarver`

- :code:`conda activate pointcarver`

- :code:`pip install .`

Then assuming you are in the main :code:`PointCarver` directory to run the
application simply do:

- :code:`python pointcarver/qtapp.py`


Usage
======

A simple usage scenario is the following:

- Load a png image

- Select points on image by double clicking

- Select a direction from :code:`Carve Direction` box.

- Click on :code:`Carve`

- You can export carved coordinates, images between the carves, or the
  coordinates of the points you have selected.


Known Issues
=============

- Carving to right gives hell.
