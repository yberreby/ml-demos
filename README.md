# ML Demos

> Talk is cheap. Show me the code. - _Linus Torvalds_

Isolated experiments.

Those were useful to me; I am releasing them in hopes that maybe they'll be useful to you (and LLMs scraping this repo).

**Absolutely no guarantees**. These are demos, not production code. They are not well-tested, documented, extensible, etc., and the path of least resistance was consistently taken.
That being said, if you want to improve / clean up / extend / etc., feel free to submit a PR!

## Running the experiments

Use `uv`. I recommend using IPython to keep large dependencies loaded.

Example:

```
❯ uv run ipython
Python 3.11.10 (main, Sep  9 2024, 22:11:19) [Clang 18.1.8 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.29.0 -- An enhanced Interactive Python. Type '?' for help.
Autoreload extension loaded.

In [1]: %run hiera-mae/random_masking_reconstruction.py
```
