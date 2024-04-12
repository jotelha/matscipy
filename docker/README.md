# Docker container for matscipy on fencisx

Buil the container with 

    docker build docker -t fenics-matscipy

from repository root and run with

    docker run -v $(pwd):/tmp/matscipy --init -ti -p 8888:8888 fenics-matscipy

From within `/tmp/matscipy` within the container, run

    git config --global --add safe.directory /tmp/matscipy

and

   python3 -m pip install --no-build-isolation --editable .[test]

to install the current state of th repository editable upon the pre-installed matscipy.
