
# The phonology of sperm whale coda vowels

The data files in this [OSF deposit](https://osf.io/9t6qu)
accompany the article [The phonology of sperm whale coda vowels]().
The following sections describe the columns of the `.csv` files by label.

## `codamd.csv`

The `codamd.csv` file contains metadata relating to codas and
coda vowel types as determined by a human annotator.

  * `codanum`: The unique identifier assigned to a recorded coda.
  * `focal`: Boolean value that indicates whether the recorded coda was created by the tagged whale (True) or another whale (False).
  * `whale`: The name of whale that produced the recorded coda. Possible values are ['ATWOOD', 'FORK', 'FRUIT', 'JOCASTA', 'LADYO', 'LAIUS', 'NALGENE', 'PINCHY', 'SAM', 'SOPH', 'SOURSOP', 'TBB', 'TWEAK'] and an empty value means the whale was not identified.
  * `codatype`: The traditional timing-based coda type. Possible values are ['1+1+3', '1+31', '1-NOISE', '10-NOISE', '10R', '10i', '2+3', '2-NOISE', '3-NOISE', '3D', '3R', '4-NOISE', '4R2', '5-NOISE', '5R1', '5R2', '5R3', '6-NOISE', '6R', '6i', '7-NOISE', '7D2', '7i', '8-NOISE', '8D', '8R', '8i', '9-NOISE', '9R', '9i'].
  * `Duration`: The time in seconds of the coda, calculated as the sum of all the inter-click intervals (ICI) of the coda.
  * `handv`: The spectral-based coda vowel type as manually assigned by a human evaluator. Possible values are ['a', 'i'] and an empty value means the annotator could not determine the coda vowel type.


## `codasp.csv`

The `codasp.csv` file contains results from spectral analysis of the coda
clicks.

  * `codanum`: The unique identifier assigned to a recorded coda.
  * `autovpkcodastr`: String composed of sequences of 'a' and 'i' for each click in the coda, as assigned by automated processing.
  * `autovbycoda`: Coda vowel category as assigned by automated processing. If all clicks in the coda were assigned the same 'a' or 'i' value, then the corresponding value is assigned here. If there are any mismatched clicks in the coda, then '?' is assigned.
  * `mismatchN`: The number of mismatched clicks in the coda. These are the clicks that were not assigned the same 'a' or 'i' value as the most common value in the coda.
  * `mismatchpct`: The percentage of mismatched clicks in the coda. These are the clicks that were not assigned the same 'a' or 'i' value as the most common value in the coda. 
  * `meandist_pk1`: The mean distance in Hz of the first spectral peak in each second and later click in a coda from the first spectral peak of the first click.
  * `meandist_pk2`: The mean distance in Hz of the second spectral peak in each second and later click in a coda from the second spectral peak of the first click.
  * `meandiff_pk1`: The mean distance in Hz of the first spectral peak in each second and later click in a coda from the first spectral peak of its preceding click.
  * `meandiff_pk2`: The mean distance in Hz of the second spectral peak in each second and later click in a coda from the second spectral peak of its preceding click.

In addition to the columns described above, the `codasp.csv` file contains
the following columns repeated from `codamd.csv`:

  * `focal``
  * `whale`
  * `codatype`
  * `handv`


## `focal-coarticulation-metadata.csv`

The `focal-coarticulation-metadata.csv` file contains rows describing sequences
of clicks from consecutive codas produced by a single focal whale--the last
click from the first coda (click1) and the first click from the second coda
(click2).

  * `codanum`: The unique identifier assigned to the second coda in each sequence.
  * `prevcodanum`: The unique identifier assigned to the first coda in each sequence.
  * `whale`: The name of whale that created the consecutive coda clicks. (Repeated from `codamd.csv`.)
  * `handvcat`: Coda vowel category as manually assigned by a human evaluator for the second coda. (Same as `handv` of `codamd.csv`.)
  * `prevhandvcat`: Coda vowel category as manually assigned by a human evaluator for the first coda.
  * `deltasec`: Time in seconds between the clicks.
  * `numpks`: The number of peaks found in the click spectrum for click2.
  * `prevnumpks`: The number of peaks found in the click spectrum for click2.
  * `coart`: Sequence of coda vowel categories as manually assigned by a human evaluator for the coda sequence. Possible values are ['aa', 'ai', 'ia', 'ii'].

  * `prevspecpk`: The frequency in Hz of the highest peak of the spectrum of click1. 
  * `specpk`: The frequency in Hz of the highest peak of the spectrum of click2. 
  * `prevpkfq`: The frequencies in Hz of the highest peaks of the spectrum of click1.
  * `pkfq`: The frequencies in Hz of the highest peaks of the spectrum of click2.
  * `prevf1pk`: The frequency in Hz of the first spectral peak of click1.
  * `f1pk`: The frequency in Hz of the first spectral peak of click2.
  * `prevf2pk`: The frequency in Hz of the second spectral peak of click1.
  * `f2pk`: The frequency in Hz of the second spectral peak of click2.

The following columns were used during processing to track corresponding parts
of data structures that are not included in the public dataset. They are briefly
described but are unlikely to be useful.

  * `audidx`: The index of the row corresponding to click2 of a 2d array of audio data.
  * `prevaudidx`: The index of the row corresponding to click1 of a 2d array of audio data.
  * `pairt1`: The start time of a pair of consecutive clicks in an output audio file.
  * `midt`: The midpoint of a pair of consecutive clicks in an output audio file.
  * `pairt2`: The end time of a pair of consecutive clicks in an output audio file.
  * `specidx`: The index of the row corresponding to click2 of a 2d array of spectral data.
  * `prevspecidx`: The index of the row corresponding to click1 of a 2d array of spectral data.
  * `prevpkidx`: The indexes into an array of spectral values where first and seconds peaks were found for click1.
  * `pkidx`: The indexes into an array of spectral values where first and seconds peaks were found for click2.

Other columns used during data processing that are not used in the included
scripts and are unlikely to be of interest:

  * `handvsubtype`: A descriptive qualifier of coda vowel category used by the manual annotator during preliminary work. The values are not well defined and are not used in the analysis.
  * `prevclicknum`: The click number (position) of click1 in its coda.
  * `clicknum`: The click number (position) of the first click in the second coda of the sequence. Always `1`.
  * `prevtagondt`: The datetime of when the DTAG was placed on the whale for the recording pertaining to the first coda in a sequence.
  * `tagondt`: The datetime of when the DTAG was placed on the whale for the recording pertaining to the second coda in a sequence.
  * `prevcodadt`: The datetime of the first click of the first coda in a sequence.
  * `codadt`: The datetime of the first click of the second coda in a sequence.
  * `prevcodaenddt`: The datetime of the last click of the first coda in a sequence.
  * `codaenddt`: The datetime of the last click of the second coda in a sequence.

