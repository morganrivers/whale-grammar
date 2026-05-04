"""whale-grammar end-to-end pipeline.

  fetch unified corpus
        |
        v
  B_classify (segment + label)
        |
        +-> data/classified/codas_classified.csv (canonical artefact)
        |   data/classified/rhythm_class_index.csv (int -> label)
        |
        +-> C_render_readable
        |       -> data/readable/whale_dialogues.txt (human transcript)
        |
        +-> E_render_csv
                -> data/classified/whale_dialogues.csv (transformer input)

Without ``--refresh``, classification is reused from the on-disk
artefact. With ``--refresh``, upstream files re-download and the
classifier re-runs.
"""
import argparse

import pandas as pd

from . import A_load_unified, B_classify, C_render_readable, E_render_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-download whale-ici-data files and re-run the classifier "
             "(otherwise the existing classified CSV is reused if present).",
    )
    args = parser.parse_args()

    if args.refresh or not B_classify.CLASSIFIED_CSV.exists():
        df = B_classify.main(refresh=args.refresh)
    else:
        print(f"D_run: reusing "
              f"{B_classify.CLASSIFIED_CSV.relative_to(B_classify.REPO)}; "
              f"pass --refresh to rebuild.")
        df = pd.read_csv(B_classify.CLASSIFIED_CSV, low_memory=False)

    C_render_readable.render(df, C_render_readable.OUT)
    print(f"D_run: rendered transcript -> "
          f"{C_render_readable.OUT.relative_to(C_render_readable.REPO)}")

    csv_out = E_render_csv.render(df)
    E_render_csv.OUT.parent.mkdir(parents=True, exist_ok=True)
    csv_out.to_csv(E_render_csv.OUT, index=False)
    print(f"D_run: wrote transformer CSV -> "
          f"{E_render_csv.OUT.relative_to(E_render_csv.REPO)} "
          f"({len(csv_out):,} rows, "
          f"{csv_out['sequenceId'].nunique()} sequences)")


if __name__ == "__main__":
    main()
