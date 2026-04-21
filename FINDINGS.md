# Analysis Findings

> Results from running the full `unwrapped` pipeline on **89,583 cleaned tracks** across **113 genres**.
> Raw dataset: 114,000 rows; 24,417 removed during cleaning (21%).

---

## 1. Which audio features are most associated with popularity?

All correlations between individual audio features and popularity are weak (|r| < 0.15), which is itself a meaningful finding: **no single audio feature is a reliable predictor of whether a track will be popular.**

| Feature | Correlation with Popularity | Direction |
|---|---|---|
| Instrumentalness | −0.13 | negative |
| Loudness | +0.07 | positive |
| Danceability | +0.07 | positive |
| Speechiness | −0.05 | negative |
| Acousticness | −0.04 | negative |
| Liveness | −0.01 | negative |
| Energy | +0.01 | positive |
| Valence | −0.01 | negative |
| Tempo | +0.01 | positive |

**Key takeaways:**

- **Instrumentalness is the strongest (negative) predictor.** Tracks with near-zero instrumentalness (i.e., vocal tracks) average 34.6 popularity, while highly instrumental tracks (0.8–1.0) average only 27.6. The bucket analysis shows this effect is monotonic — each step up in instrumentalness corresponds to a step down in popularity. This likely reflects streaming platform behavior: listeners gravitate toward vocal music, and algorithmic recommendations amplify that preference.

- **Danceability and loudness have modest positive associations.** The danceability relationship is non-linear: popularity peaks in the 0.42–0.80 range (avg ~34.3) and actually drops off for the most danceable tracks (0.80+, avg 32.5). Tracks that are *too* dance-focused may be niche.

- **Energy follows an inverted-U pattern.** The sweet spot is 0.40–0.80 energy (avg ~35.2), with both very low-energy and very high-energy tracks being less popular. Extremely high-energy genres like grindcore and heavy metal cluster at the top of the energy scale but are among the least popular genres overall.

- **Sadder and more acoustic tracks tend to be less popular on average**, but the effect is small. The "sad" genre actually bucked this trend (avg popularity 50.7), suggesting that genre identity and audience size matter more than individual audio features.

- **Explicit tracks are slightly more popular** (mean 36.9 vs 32.9 for non-explicit), which likely reflects the demographics of heavy streaming users rather than a causal audio quality effect.

---

## 2. How do genres differ in their audio profiles?

### Most vs. Least Popular Genres

**Top 5 by average popularity:**

| Genre | Avg Popularity | Notable Audio Profile |
|---|---|---|
| K-pop | 59.4 | Balanced energy (0.68), low instrumentalness, high loudness |
| Pop-film | 59.1 | Moderate energy, notably acoustic (0.45) |
| Metal | 56.7 | Very high energy (0.84), loud (−4.9 dB) |
| Chill | 54.1 | Low energy (0.43), high acousticness (0.53) |
| Latino | 51.8 | Highest danceability among top genres (0.76) |

**Bottom 5 by average popularity:**

| Genre | Avg Popularity | Notable Audio Profile |
|---|---|---|
| Iranian | 2.2 | Very low valence (0.15), high instrumentalness (0.59) |
| Romance | 3.5 | Extremely acoustic (0.86), very low energy |
| Latin | 9.6 | Danceable (0.72), energetic — but low popularity despite it |
| Jazz | 9.7 | High acousticness (0.77), low energy (0.31) |
| Detroit Techno | 11.1 | High instrumentalness (0.70), danceable but niche |

**Interesting contrasts:**

- **"Latino" vs "Latin"** — these are two distinct genres in the dataset with wildly different popularity (51.8 vs 9.6) despite similar audio profiles (both danceable, energetic, low acousticness). This is a labeling artifact worth flagging: "Latin" in the dataset appears to capture older or more regional Latin music, while "Latino" skews toward modern urban hits.

- **K-pop's dominance** (avg 59.4) is striking given that it's a non-English-language genre. It has above-average danceability, controlled loudness, and very low instrumentalness — essentially a maximally listener-optimized audio profile. It also benefits from a dedicated global fanbase that drives streaming numbers.

- **Classical and jazz are penalized by instrumentalness.** Both genres have high instrumentalness (classical: 0.60, jazz: 0.11) and are among the least popular. This is consistent with the feature-level finding above.

- **The "chill" genre challenges the energy-popularity assumption.** Despite below-average energy (0.43), chill has the 4th-highest avg popularity (54.1). This suggests that genre-level audience size and platform curation matter more than raw audio energy.

- **EDM genres split dramatically.** Progressive house (avg 48.3) significantly outperforms detroit-techno (11.1) and chicago-house (12.3), despite similar audio signatures. The difference likely lies in mainstream crossover appeal — progressive house has had major pop crossovers; the classic house and techno genres remain more underground.

---

## 3. Which tracks are outliers within their genre?

The outlier detection computes z-scores for each track's audio features relative to its genre's mean and standard deviation, then flags tracks with a combined deviation score above 2.0. Twenty tracks cleared this threshold.

**Most notable outliers:**

| Track | Artist | Genre | Deviation Score | Why It's Unusual |
|---|---|---|---|---|
| Varraaru Vaarraaru Yaaru Varraaru | Mohan; Murali; Chandiran | Ambient | 4.31 | High energy, vocal-heavy — nothing like ambient's quiet, instrumental norm |
| Some Broken Hearts Never Mend | Don Williams | Country | 4.04 | Very low energy and tempo relative to country's upbeat profile |
| Fried Noodles (Getter Remix) | Pink Guy; Getter | Comedy | 3.92 | Heavy electronic production in a genre known for low-fi novelty tracks |
| Blue Side (Outro) | j-hope | K-pop | 2.87 | Stripped-down outro track — unusually low energy and instrumentalness for K-pop |
| Soldier of Fortune | Opeth | Death Metal | 2.82 | Acoustic, low-energy ballad in an otherwise extreme genre |

**What the outlier distribution tells us:**

- Outliers are spread across genres rather than concentrated in one — no single genre is systematically "noisy."
- Several outliers are intentional artistic departures (Opeth's acoustic track in death metal, j-hope's quiet outro), rather than mislabeled data.
- The comedy genre appears twice, suggesting its audio profile is highly variable — comedy tracks range from lo-fi spoken word to full electronic production.
- Deviation scores drop off quickly after the top few outliers: 16 of the 20 tracks cluster between 2.8 and 3.5, indicating that extreme outliers are rare. The vast majority of tracks sit comfortably within their genre's audio norms.

---

## 4. What does the outlier distribution look like across features?

The IQR-based outlier analysis reveals that **some features are far noisier than others:**

| Feature | Outlier Rate | Interpretation |
|---|---|---|
| Instrumentalness | 22.2% | Extremely bimodal — most tracks are fully vocal or fully instrumental, few in between |
| Speechiness | 11.6% | Most tracks have very low speechiness; rap/spoken word creates a long right tail |
| Time Signature | 10.7% | Nearly all tracks are in 4/4; anything else is flagged as an outlier |
| Liveness | 7.6% | Studio recordings cluster tightly; live recordings pull the right tail |
| Loudness | 5.4% | A few very quiet tracks (classical, ambient) create a long left tail |
| Duration | 4.9% | Some tracks are extremely long (DJ sets, classical movements) |

The instrumentalness distribution is the most striking: its median is essentially 0 (4.2×10⁻⁵), meaning more than half the dataset has near-zero instrumentalness. But the mean is 0.156, pulled up by a cluster of fully instrumental tracks. This bimodality means the IQR is very narrow, making almost any moderately instrumental track technically an "outlier" — the method flags 22% of the dataset. For analysis purposes, instrumentalness should be treated as a near-binary feature rather than a continuous one.

---

## 5. What is the overall data quality situation?

The dataset is in good shape, with a few caveats:

- **Missing values are minimal.** Only 3 columns have any nulls at all (artists, album_name, track_name — 1 row each, 0.001%). This is unusually clean for a real-world dataset.

- **The cleaning pipeline removed 21% of rows (24,417).** The primary culprits are likely duplicate track IDs across genre assignments (the same track appears in multiple genre buckets in the raw export) and rows failing range validation. The deduplication step — keeping the most complete row per track_id — is the most impactful cleaning step.

- **Popularity is well-distributed** (mean 33.2, median 35.0, roughly symmetric) with only 2 IQR outliers. This suggests the 0–100 scale is being used across its full range, though a small cluster of tracks with popularity = 0 may represent delisted or very obscure tracks rather than truly zero-popularity songs.

- **Duration has extreme outliers** (max 5,237 seconds — nearly 90 minutes). These are almost certainly DJ mixes, live recordings, or classical works that were not filtered by the cleaning pipeline. Downstream models that use duration as a feature may want to cap it.

- **Genre labeling is the weakest dimension.** The "latino" vs "latin" discrepancy noted above, and the presence of genres like "study," "sleep," and "sad" (which are mood/use-case labels rather than musical genres) indicate that the genre taxonomy mixes multiple classification schemes. Cross-genre comparisons should be interpreted with this in mind.
