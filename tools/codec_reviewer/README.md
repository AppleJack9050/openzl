# Codec reviewer

Review the codec graph that `zli compress` actually ran, from its trace.

A trace of a real file records thousands of codec nodes, but most of them are
the same few pipelines repeated for every column, field or chunk. The reviewer
groups those repeats and shows each distinct pipeline once, with its sizes
summed over all runs. You can see where the compressed bytes go, which codecs
and settings each part of the input got, which streams were merged, and whether
anything failed.

It needs only the Python 3 standard library.

## Record a trace

```bash
zli compress FILE -p parquet -o FILE.zl --trace FILE.cbor > FILE.dot
```

`--trace` writes the CBOR file and also prints the same graph as DOT text to
stdout. The reviewer reads either one, or either one gzip-compressed.

## Review it

```bash
tools/codec_reviewer/codec_reviewer.py FILE.cbor                  # summary
tools/codec_reviewer/codec_reviewer.py FILE.cbor --show M1 --show S2   # pipelines as trees
tools/codec_reviewer/codec_reviewer.py FILE.cbor --chunk 3        # one chunk of a segmented trace
tools/codec_reviewer/codec_reviewer.py FILE.cbor --quiet --html FILE.review.html
tools/codec_reviewer/codec_reviewer.py --serve 8765 [FILE.cbor ...]  # reviews in the browser
```

The summary lists every distinct pipeline with its compressed bytes, ratio,
codecs per run and number of runs, followed by the codec census:

```
Codec review: sensors_chunks.cbor.gz
CBOR (trace version 1) · all 4 chunks
2,889,011 B in → 560,780 B in streams (5.15×) · 356 codecs ran · 22 distinct pipelines
Coverage OK: 595 of 595 codec nodes, 560,435 of 560,435 stored bytes, 36 of 36 split outputs, 8 of 8 merges

MERGES (8)
  ID          Bytes    Ratio Codecs  Runs  Pipeline
  M1        142,670    2.23×     23     1  concat_num of 2 streams: from outputs #7, #8 of dispatchN_byTag
                                             concat_num → field_lz → quantize_offsets → fse_v2 → fse_ncount
```

`--show ID` draws one pipeline. Byte counts on the right are what each stream
costs, including everything downstream of it. Output numbers skip streams that
went straight to the frame; those are counted in "writes":

```
M1  concat_num of 2 streams — from outputs #7, #8 of dispatchN_byTag
    318,216 raw B → 142,670 B · 2.23× · 23 codecs per run · ran once · chunk 3

2 streams in (318,216 B)                                                  142,670 B
└── concat_num  graph cluster · config size 76                            142,670 B
    ├── #0 convert_num_to_serial_le  writes 8 B · header 1 B                    9 B
    └── #1 field_lz (via convert_num_to_struct_le)  header 3 B            142,661 B
        ├── #0 transpose_split  graph field_lz_literals · writes 4,102 B    6,160 B
        │   ├── #1 bitpack_serial  writes 2,051 B · header 1 B              2,052 B
        ...
        ├── #2 quantize_offsets  graph field_lz · writes 99,788 B         126,452 B
        │   └── #0 fse_v2 (via convert_num_to_serial_le)  graph fse · writes 26,639 B · header 4 B  26,664 B
        ...
```

`--html` writes a self-contained page with the same analysis: a pipeline list
with a filter, drawn pipelines (line thickness is compressed bytes), a chunk
selector and the codec census. It loads nothing from the network. In a drawn
pipeline, the mouse wheel zooms around the pointer and dragging moves the
drawing. The zoom bar (or `+`, `-`, `0` and `F` with the drawing focused) zooms
in, zooms out, returns to 100% or fits the whole pipeline.

| Option | Effect |
| --- | --- |
| `--show ID` | Print pipeline `ID` as a tree. Repeat it, pass `M1,S2`, or pass `all`. |
| `--show-conversions` | Keep `convert_*` codecs in trees instead of folding them into the next step. |
| `--chunk N` | Review one chunk: a chunk number, `top` for the top-level segmenter run, or `all` (default). |
| `--limit N` | Rows per summary section, 15 by default; `0` shows everything. |
| `--html OUT`, `--json OUT` | Also write the interactive report, or the analysis as compact JSON. |
| `--serve PORT` | Serve reviews in the browser until Ctrl+C, starting with the traces given (any number, or none); `0` picks a free port. If a reviewer already runs on `PORT`, add the traces to it instead. See [Serve reviews](#serve-reviews). |
| `--host ADDR` | Address for `--serve`. The default, `127.0.0.1`, accepts connections only from the same machine. |
| `--quiet` | Print only `--show` trees. |
| `--ascii` | Draw with ASCII only. This is automatic when stdout can't encode Unicode. |
| `--fail-on-incomplete` | Exit 3 if the coverage check doesn't add up. |

Exit status is 0 on success, and 2 in these cases:
- the trace can't be read;
- an option names a chunk or pipeline that doesn't exist;
- an output file can't be written, or `--serve` can't use its port;
- a trace couldn't be added to a reviewer that is already running.

## Serve reviews

```bash
tools/codec_reviewer/codec_reviewer.py --serve 8765                         # start empty
tools/codec_reviewer/codec_reviewer.py --serve 8765 a.cbor b.dot.gz         # start with two reviews
tools/codec_reviewer/codec_reviewer.py --serve 8765 c.cbor                  # later: add c.cbor to it
```

Open `http://127.0.0.1:8765/`. On a remote machine, forward the port first, for
example with VS Code's Ports view or `ssh -L 8765:127.0.0.1:8765 HOST`.

- **Open trace…** or dropping trace files anywhere on the page sends them to
  the server. The server reviews them one at a time with the same code as the
  command line, then opens the last one. Files dropped while a review runs
  wait their turn.
- Each review has its own address (`/r/1`, `/r/2`, ...), and `/` opens the
  newest one. **Reviews** lists them all as links; the list refreshes when you
  return to the tab, so traces added from the command line show up.
- Traces given on the command line are reviewed first, with the other options
  applied (`--show`, `--html`, `--chunk`, ...). If one of them fails, nothing is
  served or added. If a reviewer already serves on the port, the command adds
  the traces to that reviewer and prints their addresses instead of starting a
  second server.
- The server keeps the newest 50 reviews in memory and accepts uploads up to
  512 MiB. Reviews are gone when it stops, and numbering starts again at 1, so
  an old address may show "There is no review N".
- Any trace is refused if it is larger than 1 GiB after gzip decompression, on
  the command line too.

The server is meant for your own machine. On the default address it answers only
requests addressed to `localhost`, `127.0.0.1` or `[::1]`, on any port, so a
forwarded port works and other websites can't reach it by DNS rebinding. Traces
can be opened only from the review page itself (the browser vouches for that
with `Sec-Fetch-Site`, so a proxy that rewrites the Host header still works), or
by a client that isn't a browser, such as this command. Anyone who can connect to the port can still read
the reviews: other users of a shared machine can reach `127.0.0.1` too, and
`--host 0.0.0.0` opens the port to the network.

## How pipelines are formed

The trace is a DAG of codecs and streams. A segmented compression has one graph
for the top level (`#start → segmenter`) and one per chunk.

1. **Cuts.** The graph is cut at *merges*, meaning codecs with more than one
   input such as the `concat_num` that joins a cluster of columns (a codec that
   every input abandoned is an attempt instead; see Failures). It is also
   cut at *splits*: `dispatch*` codecs with at least 3 outputs that continue to
   other codecs (one per Parquet column or CSV field), and any other codec with
   at least 24 such outputs. `transpose*` codecs, which make one stream per
   byte of a record, are never cut; neither is a narrow `splitN`.
2. **Signatures.** Each cut piece gets a structural signature. It covers:
   - codec and graph names (without the `#N` instance suffix) and graph types
   - int parameters, and the ids of copy and ref parameters
   - failures, abandoned attempts and fan-in
   - for each output, in output-index order, where it ends: another codec, the
     frame (`store`), a merge, or a split
3. **Groups.** Pieces with equal signatures, cut at the same kind of codec,
   form one pipeline, and their sizes are summed.

IDs are `E` for entry pipelines (starting at the input), `M` for merges and `S`
for split outputs, numbered by compressed bytes within each kind. A split
output whose only job is to reach a merge is listed separately. Its bytes are
counted in the merge.

Sizes follow the tracer (`ChunkTrace::fillCSize`): a stream's size is its
consumer's header plus the sizes of that codec's outputs, and a merge's cost is
split evenly across its inputs. "Writes" are streams handed to `store`.
"Bytes in streams" excludes frame headers, so it is slightly below the size of
the `.zl` file.

Every analysis checks that each codec node, stored byte, split output and merge
appears in exactly one pipeline. The summary and the HTML footer report that
check.

### Failures

The tracer keeps failed work in the trace, and the reviewer shows it where it
happened:

- **A failed codec** is marked FAILED, with its error underneath.
- **A graph that fails** reports its error once, on its first codec. If it
  failed before running any codec, the tracer adds a `#in_progress`
  placeholder, which carries the error instead.
- **Abandoned consumers** are listed before the consumer that finally took the
  stream, marked "(abandoned)". This covers a failed codec, a failed graph's
  placeholder, and work that permissive mode rolled back, such as a conversion
  whose next codec failed. A codec with several inputs is listed under its
  first input. Its codec nodes count toward coverage. Its stores do not count
  as stored bytes, because rolled-back work never reaches the frame, and its
  streams are drawn without sizes.
- **Kept work** is what you reach from the input by following each stream to
  its final consumer. A kept stream whose final consumer failed or is a
  placeholder shows as "never compressed". A failed segmenter is one such
  consumer; so is a codec that fails in `--strict` mode.
- **Compression failed** when such a stream exists. No ratio is shown then.
  Pipelines that contain the failure have no size, and the summary gives only
  the stream bytes of the chunks that finished, or "nothing written".

Only the CBOR trace carries error messages; the DOT text just marks that
something failed.

## Tests

```bash
make codec_reviewer_test
# or: python3 -m unittest discover -s tools/codec_reviewer/tests
```

`tests/trace_builder.py` builds zli-shaped traces in memory and emits them as
CBOR and as DOT. The files in `tests/data` are real zli traces:

- `sensors_chunks.*`: a synthetic 2.9 MB sensor table (timestamps, ids,
  temperatures, a status string, a list column) written as uncompressed PLAIN
  Parquet. Recorded with `zli compress sensors.parquet -p parquet
  --chunk-size-mb 1 -o sensors.zl --trace sensors_chunks.cbor >
  sensors_chunks.dot`, then gzipped.
- `serial.*`: the same file with `-p serial`.
- `compressed_parquet_failure.cbor`: `-p parquet` on a Snappy-compressed copy,
  which the Parquet profile rejects.
