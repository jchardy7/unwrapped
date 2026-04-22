"""
demo_preference.py
------------------
Demonstrates the LikedSongs preference engine, which scores every track in
the dataset by audio-feature similarity to a user-defined liked-songs list.

The demo dynamically picks 5 of the most-popular tracks as a seed list so it
works on any version of the dataset without hardcoding IDs.

Run from the project root:
    python scripts/demo_preference.py
"""

from unwrapped.clean import clean_data
from unwrapped.io import load_data
from unwrapped.preference import LikedSongs


DATA_PATH = "data/raw/spotify_data.csv"
N_SEED_SONGS = 5
N_RECOMMENDATIONS = 10


def main() -> None:
    print("=" * 60)
    print("LikedSongs Preference Engine — STAT 3250 Final Project")
    print("=" * 60)

    print("\nLoading and cleaning data...")
    raw_df = load_data(DATA_PATH)
    df, _ = clean_data(raw_df)
    print(f"  {len(df):,} tracks available")

    # Seed the liked list with the 5 most popular tracks in the cleaned dataset
    top_tracks = (
        df[["track_id", "track_name", "artists", "track_genre", "popularity"]]
        .sort_values("popularity", ascending=False)
        .drop_duplicates(subset="track_id")
        .head(N_SEED_SONGS)
    )

    liked = LikedSongs(df)
    print(f"\nAdding {N_SEED_SONGS} seed tracks (top by popularity):")
    for _, row in top_tracks.iterrows():
        liked.add_by_id(row["track_id"])
        print(f"  + {row['track_name']} — {row['artists']}  (popularity {row['popularity']})")

    # Build and display the taste profile
    profile = liked.build_profile()
    print("\n--- Your Taste Profile (mean audio features of liked songs) ---")
    for feature, value in profile.items():
        print(f"  {feature:<22} {value:.3f}")

    # Get top recommendations
    print(f"\n--- Top {N_RECOMMENDATIONS} Recommended Tracks ---")
    recs = liked.predict(top_n=N_RECOMMENDATIONS, exclude_liked=True)
    for rank, (_, row) in enumerate(recs.iterrows(), start=1):
        print(
            f"  {rank:>2}. {row['track_name']:<35}  {row['artists']:<25}  "
            f"genre={row['track_genre']:<18}  score={row['preference_score']:.4f}"
        )

    # Show a feature-level explanation for the top recommendation
    top_id = recs.iloc[0]["track_id"]
    top_name = recs.iloc[0]["track_name"]
    print(f"\n--- Why '{top_name}' scored highest (top matching features) ---")
    try:
        explanation = liked.explain_top(top_id, n=3)
        matches = explanation.get("matches")
        mismatches = explanation.get("mismatches")

        if matches is not None and not matches.empty:
            print("  Best matches (closest to your taste profile):")
            for _, item in matches.iterrows():
                print(f"    {item['feature']:<22}  profile={item['profile_raw']:.3f}  track={item['track_raw']:.3f}")
        if mismatches is not None and not mismatches.empty:
            print("  Biggest gaps (furthest from your taste profile):")
            for _, item in mismatches.iterrows():
                print(f"    {item['feature']:<22}  profile={item['profile_raw']:.3f}  track={item['track_raw']:.3f}")
    except Exception:
        print("  (explanation not available for this track)")


if __name__ == "__main__":
    main()
