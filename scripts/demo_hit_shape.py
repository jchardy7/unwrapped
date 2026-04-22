"""
demo_hit_shape.py
-----------------
Demonstrates the hit shape predictor pipeline, which classifies tracks as
"hits" (popularity >= 70) or "non-hits" using audio feature similarity to
hit/non-hit centroids, then trains logistic regression and random forest
classifiers.

Run from the project root:
    python scripts/demo_hit_shape.py
"""

from unwrapped.hit_shape_predictor import run_hit_shape_pipeline


DATA_PATH = "data/raw/spotify_data.csv"
HIT_THRESHOLD = 70


def main() -> None:
    print("=" * 60)
    print("Hit Shape Predictor — STAT 3250 Final Project")
    print("=" * 60)

    print(f"\nRunning pipeline (hit threshold = popularity >= {HIT_THRESHOLD})...")
    results = run_hit_shape_pipeline(
        data_path=DATA_PATH,
        hit_threshold=HIT_THRESHOLD,
        save_results=False,
    )

    # --- Hit vs non-hit audio profiles ---
    profiles = results["profiles"]
    print("\n--- Audio Feature Profiles (mean values) ---")
    print(profiles.to_string())

    # --- Biggest differences between hit and non-hit profiles ---
    diffs = results["differences"]
    print("\n--- Top Features That Separate Hits from Non-Hits ---")
    top_diffs = diffs.sort_values("absolute_difference", ascending=False).head(5)
    for _, row in top_diffs.iterrows():
        direction = "higher in hits" if row["hit_mean"] > row["non_hit_mean"] else "lower in hits"
        print(
            f"  {row['feature']:<22}  "
            f"hit={row['hit_mean']:.3f}  non-hit={row['non_hit_mean']:.3f}  "
            f"({direction})"
        )

    # --- Model comparison ---
    comparison = results["comparison"]
    print("\n--- Model Comparison ---")
    display_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1"]
                    if c in comparison.columns]
    print(comparison[display_cols].to_string(index=False))

    # --- Feature importance ---
    importance = results["feature_importance"]
    print("\n--- Top Feature Importances (Random Forest) ---")
    print(importance.head(5).to_string(index=False))

    # --- Summary stats on test predictions ---
    preds = results["predictions"]
    total = len(preds)
    actual_hits = int(preds["actual_is_hit"].sum())
    predicted_hits = int(preds["predicted_is_hit"].sum())
    print(f"\n--- Test Set Summary ---")
    print(f"  Total test tracks : {total:,}")
    print(f"  Actual hits       : {actual_hits:,}  ({actual_hits / total:.1%})")
    print(f"  Predicted hits    : {predicted_hits:,}  ({predicted_hits / total:.1%})")


if __name__ == "__main__":
    main()
