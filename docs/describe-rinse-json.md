# RINSE records from `occ describe`

`occ describe` computes the RINSE crystal descriptor for CIF and SHELX
(`.res`/`.ins`) structures. With `--json` or `--jsonl` it writes one record per
structure for other programs to read:

```bash
occ describe landscape.cif --json rinse.json
occ describe *.cif *.res --jsonl rinse.jsonl --hash-words 2
```

Every data block of a multi-block CIF is a structure of its own. A block or file
that cannot be described, such as a data block with no unit cell, is still a
record, carrying an `error` in place of `values` and `hash`, so the records line
up with the input. Writing records restricts the run to RINSE, and the command
only fails if no structure at all could be described.

## Missing occupancies and displacement parameters

A SHELX atom line without an occupancy is fully occupied, and one without a
displacement parameter has none (U = 0), as do CIF atoms without ADPs.
rinse-descriptor 3.2.0 instead fills in U = 0.1 Å², and an occupancy of 0.5,
so the two differ for such files. To compare structures that do not all
carry ADPs, such as predicted structures against experimental ones, give every
atom the same isotropic U with `--fixed-uiso`.

## `--json`

One document:

```json
{
  "occ_version": "0.9.7",
  "descriptor": "rinse",
  "rinse_version": "3.2.0",
  "params": {
    "n_max": 16, "l_min": 4, "l_max": 20, "l": [4, 6, 8, 10, 12, 14, 16, 18],
    "radial_scale": 0.3, "qmax_factor": 1.2, "radial_basis": "cv_gaussian",
    "monopole_normalisation": true, "log1p": false, "normalisation": "l2",
    "fixed_uiso": null, "hash_words": 1, "distance": "euclidean"
  },
  "shape": [16, 8],
  "layout": "row-major [radial][angular]",
  "records": [
    {"file": "landscape.cif", "block": "s0001", "index": 0,
     "values": [0.0216, 0.0235, ...], "hash": "fokif", "error": null},
    {"file": "landscape.cif", "block": "notes", "index": 1,
     "values": null, "hash": null,
     "error": {"message": "Missing unit cell data"}}
  ]
}
```

- `rinse_version`: the [rinse-descriptor](https://github.com/DuMOCC-Group/rinse-descriptor)
  release whose descriptors and hashes occ reproduces. `params` use its
  parameter names.
- `block`: the CIF data block name, or the file name's stem for `.res`/`.ins`.
  `index` is the block's position in its file, counting from 0, since block
  names can repeat.
- `values`: the flattened descriptor, `shape[0] * shape[1]` numbers laid out
  row-major, i.e. `values[n * shape[1] + j]` is radial order `n` at angular
  level `params.l[j]`.

## `--jsonl`

The same fields as JSON Lines, written as each structure is described: the
first line is the document above without `records`, then one record per line.

## Comparing structures

- **Distance:** the descriptor is scaled to unit length
  (`"normalisation": "l2"`), so Euclidean distance between `values` is monotone
  with cosine distance (d² = 2 − 2 cos θ). Use plain Euclidean distance on
  `values`, as `rinse --compare` does by default.
- **Hash:** a SimHash over a fixed PCA model. Bit *k* is the sign of the
  descriptor's projection onto principal component *k*, sixteen bits to a
  five-letter word, the first word holding the leading components. Structures
  sharing a one-word hash fall in the same orthant of the top sixteen
  components, a coarse bucket; more words give finer buckets. Hashes are for
  grouping and lookup; rank by distance on `values`.
