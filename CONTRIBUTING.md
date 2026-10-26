# Contributing

This repository accompanies a research study. Please open an issue before
proposing a substantial change so that scientific and software changes can be
reviewed separately and reproduced clearly.

For a pull request:

1. create a focused branch;
2. describe whether the change affects code, results, or documentation;
3. add or update tests where appropriate;
4. run `python -m unittest discover -s tests`;
5. do not commit datasets other than the attributed HTRU2 file, binary caches,
   credentials, private documents, or machine-specific logs.

Changes that affect numerical results must use a new output directory and must
not overwrite the reference tables associated with an existing release tag.
