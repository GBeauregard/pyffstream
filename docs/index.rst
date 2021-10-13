pyffstream
==========

.. toctree::
   :hidden:
   :maxdepth: 2

   reference
   license

.. image:: https://img.shields.io/pypi/v/pyffstream.svg
   :target: https://pypi.org/project/pyffstream/
.. image:: https://img.shields.io/pypi/pyversions/pyffstream.svg
   :target: https://pypi.org/project/pyffstream/
.. image:: https://github.com/gbeauregard/pyffstream/workflows/Release/badge.svg
   :target: https://github.com/GBeauregard/pyffstream/actions/workflows/release.yml
.. image:: https://github.com/gbeauregard/pyffstream/workflows/Tox/badge.svg
   :target: https://github.com/GBeauregard/pyffstream/actions/workflows/tox.yml
.. image:: https://github.com/gbeauregard/pyffstream/workflows/CodeQL/badge.svg
   :target: https://github.com/GBeauregard/pyffstream/actions/workflows/codeql-analysis.yml
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

A CLI wrapper for ffmpeg to stream files over SRT/RTMP. Also supports
the api for a not (yet) open sourced pyffserver endpoint.


Installation
------------

To install pyffstream download a binary from `Github releases`_, or run
this command in your terminal:

.. _Github releases:
   https://github.com/GBeauregard/pyffstream/releases

.. code-block:: console

   $ pip install pyffstream

CLI Usage
---------

.. autoprogram:: pyffstream.cli:get_parserconfig()[0]
   :prog: pyffstream
   :groups:
