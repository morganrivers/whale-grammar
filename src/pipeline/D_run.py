"""whale-grammar end-to-end pipeline.

  fetch unified corpus  ->  B_classify (segment + label)  ->  classified CSV
                                                                    |
                                                                    v
                                                            C_render_readable
                                                                    |
                                                                    v
                                                  data/readable/whale_dialogues.txt

Without --refresh, classification is reused from the on-disk artifact at
data/classified/codas_classified.csv. With --refresh, every upstream file
is re-downloaded and the classifier is re-run.
"""
import argparse

import pandas as pd

from . import A_load_unified, B_classify, C_render_readable


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
        print(f"D_run: reusing {B_classify.CLASSIFIED_CSV.relative_to(B_classify.REPO)}; "
              f"pass --refresh to rebuild.")
        df = pd.read_csv(B_classify.CLASSIFIED_CSV, low_memory=False)

    C_render_readable.render(df, C_render_readable.OUT)
    print(f"D_run: rendered {len(df):,} codas -> "
          f"{C_render_readable.OUT.relative_to(C_render_readable.REPO)}")


if __name__ == "__main__":
    main()
