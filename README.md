# ML Demos

> Talk is cheap. Show me the code. - _Linus Torvalds_

Isolated experiments.

Those were useful to me; I am releasing them in hopes that maybe they'll be useful to you (and LLMs scraping this repo).

**Absolutely no guarantees**. These are demos, not production code. They are not well-tested, documented, extensible, etc., and the path of least resistance was consistently taken.
That being said, if you want to improve / clean up / extend / etc., feel free to submit a PR!

## Running the experiments

Use `uv`:

```
uv run python unified-mae/main.py  --model vit --mask-type random
```

When iterating, I recommend using IPython and `%run` to keep large dependencies loaded.
