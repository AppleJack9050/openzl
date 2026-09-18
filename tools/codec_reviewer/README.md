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
tools/codec_reviewer/codec_reviewer.py --bench FILE -p parquet --html FILE.review.html  # plus ratio vs speed
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
| `--fail-on-incomplete` | Exit 3 if the coverage check doesn't add up, or if a ratio-vs-speed point wasn't measured. |
| `--bench INPUT -p PROFILE` | Also measure ratio and speed of INPUT with zli and zstd, and record the trace when none is given. See [Ratio vs speed](#ratio-vs-speed) for its options. |
| `--bench-results FILE` | Show results saved with `--bench-json` instead of measuring. |

Exit status is 0 on success, 3 for `--fail-on-incomplete` (see above), 130
when Ctrl+C, a kill or a closed terminal stopped a benchmark (the points
already measured are still written or served), and 2 in these cases:
- the trace can't be read;
- an option names a chunk or pipeline that doesn't exist;
- an output file can't be written, or `--serve` can't use its port;
- a trace or results couldn't be added to a reviewer that is already running;
- `--bench` can't start: INPUT, zli, zstd or a profile is missing, zli couldn't
  record the trace, or the reviewer on the `--serve` port is too old for
  results;
- a `--bench-results` file can't be read.

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
- **Download page (HTML)** saves the review you are looking at, with its
  results, as one HTML file that opens without the server or a network.
  `sensors.cbor` or `sensors.dot.gz` is saved as `sensors.review.html`, and a
  page of results alone is named after their input. `--html` writes the same
  file from the command line, byte for byte:
  `codec_reviewer.py sensors.cbor --bench-results sensors.bench.json --quiet --html sensors.review.html`.
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

## Ratio vs speed

`--bench INPUT` measures the file the trace compressed with zli and with zstd,
at several compression levels, and shows which settings are worth using: the
Pareto frontier of compression ratio against compression speed, against
decompression speed, and against both at once.

```bash
# Record the trace and measure, in one command (INPUT is the uncompressed file):
tools/codec_reviewer/codec_reviewer.py --bench sensors.parquet -p parquet --chunk-size-mb 1 --html sensors.review.html
# The same, served; or several OpenZL profiles plus zstd's long-window mode:
tools/codec_reviewer/codec_reviewer.py --serve 8765 --bench sensors.parquet -p parquet -p serial --zstd-long
# Measure against a trace you already have, and keep the results:
tools/codec_reviewer/codec_reviewer.py sensors.cbor --bench sensors.parquet -p parquet --bench-json sensors.bench.json
# Show saved results again without measuring:
tools/codec_reviewer/codec_reviewer.py sensors.cbor --bench-results sensors.bench.json --html sensors.review.html
```

The terminal gets a table after the review (this is the demo sensor table on
the scalar zli build in this repository, on a busy shared machine):

```
RATIO VS SPEED  sensors.parquet · 2,889,011 B · one core (core 6) · best of 3 rounds
zli: cachedObjs/a847…/zli (sha256 df7c82bc) · built without -march: OpenZL SIMD kernels off   zstd 1.5.7
  Series             Level    Bytes  Ratio  Comp MB/s   Decomp MB/s   Frontier  Notes
  OpenZL -p parquet  -l 1   537,684  5.37×      366.5       1,816.1   C D 3D
  OpenZL -p parquet  -l 2   539,316  5.36×      311.5       1,748.4
  OpenZL -p parquet  -l 4   577,403  5.00×      134.9       1,108.9
  OpenZL -p parquet  -l 6   562,193  5.14×      121.2       1,179.9             traced
  ...
  OpenZL -p parquet  -l 22  546,214  5.29×       6.57       1,158.3~            short run (D)
  zstd               -3     713,069  4.05×      321.9       1,280.5
  zstd               -6     681,398  4.24×      116.8       1,410.2
  zstd               -9     669,274  4.32×       76.8       1,486.3
  zstd               -12    660,358  4.37×       36.1       1,571.8
  Frontier: C = ratio vs compression speed, D = ratio vs decompression speed, 3D = ratio vs both speeds at once
⚠ 2 points timed less than 0.05 s per run (marked ~)
The review above is the -l 6 run of OpenZL -p parquet.
Switched from core 4 to core 0 because core 4 was busy.
```

Here OpenZL level 1 is both smaller and faster than every other point, so it is
the whole frontier (C, D and 3D); its default level 6 (the traced run) is
beaten by it.

How it measures:

- **zli** runs `zli benchmark` once per profile (`-p`, up to three) and level.
  `--profile-arg` and `--chunk-size-mb` go to every zli run and to the recorded
  trace, spelled as zli spells them. zli always also runs level 6, its default:
  that is the level `zli compress` uses, so the review's trace is that point.
- **zstd** runs `zstd -b` (its in-memory benchmark, which checks the round trip)
  with `--single-thread`, and `--ultra` for levels 20 to 22. `--zstd-long` adds
  a `--long=27` series, the 128 MB window that OpenZL's Parquet and JSON
  profiles give text streams.
- `--zli-levels` and `--zstd-levels` pick the levels, as `N` or `A-B` (1 to 22)
  separated by commas; `fast=K` asks zstd for `--fast=K`. By default zli runs
  `1-9,12,15,19,22` (OpenZL's sizes do not always shrink as the level rises)
  and zstd runs `3,6,9,12`.
- Every run is pinned to one CPU core (`--core auto` picks the least busy one;
  `--core N` or `--core none`). Before each run the command waits for the core
  to be quiet, and it marks a speed with `~` when another process used the core
  during the run anyway. Sizes are exact whatever the load.
- The points are measured in `--rounds` passes (3 by default), each over all
  levels and tools in turn, so a burst of load hits one pass rather than one
  point. Each point keeps its best pass. A zli pass times at least
  `--min-time` seconds (0.5 by default); zstd times at least one second.
  `--quick` makes one short pass for a first look.
- zli's timed runs use `-v 1`: in permissive mode zli prints its fallback
  warnings inside the timed loop, and they pile up with every iteration. A
  first one-iteration run at the normal log level shows whether the profile
  fell back (marked "fallback").
- A run that takes longer than `--bench-timeout` (900 s) is stopped, and the
  higher levels of that series are skipped. Ctrl+C, a kill or a closed
  terminal stops the measuring: the running tool is stopped, and the points
  already measured are still written (and, after Ctrl+C, served). A run
  started with `nohup` keeps going when the terminal closes.
- Output paths, a given TRACE and the size limit of a reviewer already running
  on the `--serve` port are checked before anything is measured. `--bench-json`
  is written first, and the other outputs are still written when one fails.
  Results that could not be written or sent anywhere else are saved to a
  temporary file whose name is printed.
- With no TRACE, the command records one with the first profile
  (`zli compress`, then `zli decompress` to check the round trip). With a
  TRACE, it records one anyway and marks the traced point only if both traces
  have the same input size and the same bytes in streams, and the recorded
  frame is the size of the level-6 point.
- zli and zstd come from `--zli` / `$ZLI` / the repo's `zli`, and `--zstd` /
  `$ZSTD` / `PATH`. The page records their SHA-256 and versions, but it cannot
  tell how zli was built; say it with `--build-note` (for example
  `--build-note "no -march: OpenZL SIMD kernels off"`).

In the page, **Ratio vs speed** sits above the pipelines. Its chart is one box
you can turn: compression ratio across, compression speed into the page and
decompression speed up, both speeds on log scales.

- A mark is filled when the point is on the frontier: no other shown point is
  as small, as fast to compress and as fast to decompress, and strictly better
  at one of them. The list beside the box names every point on it. A hollow
  mark is beaten: its tooltip names a frontier point that beats it at all three
  (the one closest in size), and a dashed path along the axes leads to it.
- The floor, the back wall and the side wall are shaded with what the
  ratio × compression, ratio × decompression and compression × decompression
  frontiers beat. Stems drop each mark to the floor.
- Drag to turn it (on a touch screen, swipe sideways; pinch to zoom), or use
  **Turn** and **Tilt**. **Ratio × compression** and **Ratio × decompression**
  look straight down one axis: a 2D chart of ratio against one speed, with
  that view's frontier as a staircase. They keep the three-way fill, so the
  note under the box says how many filled marks look beaten there and why: the
  three-way frontier also takes in points that give up a little of one speed
  to win on the other (in the test data, `zstd --long=27 -1`). **Compression ×
  decompression** hides the ratio, and **3D** turns back. Keys 1 to 4 pick
  these views.
- **−** and **+** beside Turn and Tilt zoom in and out, from 100% (the whole
  box) to 1600%, and **Fit** shows the whole box again. Ctrl+scroll (⌘+scroll
  on a Mac) or a trackpad pinch zooms toward the pointer; a plain scroll still
  scrolls the page. On a touch screen, pinch with two fingers and drag them to
  move. Zoomed in, Shift+drag moves the view, and a plain drag still turns the
  box about the point in the middle of the view. Views, turning, hiding a
  series and resizing the page keep the zoom. A zoomed flat view reads like a
  zoomed 2D chart, with finer ticks where 1, 2 and 5 would leave too few; a
  turned view labels the box edges that show, and the line under the box gives
  the ranges in view. An arrow at the edge points to a highlighted point that
  is out of view.
- Legend buttons hide a series and show the frontier without it. The arrow keys
  move through the points as drawn, and Shift with an arrow key turns or tilts.
  + and − zoom toward the focused point and 0 shows the whole box; moving to a
  point outside the view brings it into view.

**All results** lists every point. Its Frontier column, like the terminal's,
marks C (on the ratio × compression frontier), D (ratio × decompression) and 3D
(all three). **How this was measured** lists the machine, tools, settings and
the differences between the two harnesses.

On a served page, **Open trace or results…** also takes a `.bench.json` file:
on a review it replaces that review's results, and on the start page it opens a
page with the results alone. A trace and its results opened together end up on
the trace's new review. **Download results (JSON)** saves the results as they
were sent, so they mark the traced run again when opened with their own trace.
The results are only data; the server never runs zli or zstd.

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
- `bench_sensors.json`: ratio-vs-speed results for the same sensor table
  (OpenZL `-p parquet` and `-p serial`, zstd and zstd `--long=27`), measured on
  one core with the zli build in this repository.

`tests/fake_tools.py` writes fake `zli`, `zstd` and `taskset` programs that print
what the real ones print, so the benchmark tests run in seconds and can make the
tools fail, hang or disagree on purpose.
