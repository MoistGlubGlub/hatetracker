# Hate Tracker

A python package that contains various utilities for tracking and indexing hate speech.

## Installation

To install, clone this repository and then run:

```
pip install <repository_directory>
```

To install the hatetracker package as an editable directory, run:

```
pip install -e <repository_directory>
```

Alternatively, this package can be installed directly from Github by running:

```
pip install git+https://github.com/MoistGlubGlub/hatetracker.git
```

## Utilities

### Text Rank

Run the text rank algorithm on a single document or a collection of documents, and export the results as a CSV file.
To get started, run:

```
hatetracker-rank --help
```

In general, the text rank algorithm can be run using the command


```
hatetracker-rank <args>
```
