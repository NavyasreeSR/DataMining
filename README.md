# Consumer Behavior Analytics & Digital Twin

**B565 Data Mining — Indiana University Bloomington, Spring 2024**

A consumer **digital twin**: a data-driven model of one shopper's grocery behavior, used to
predict what they will buy next and when. Built from a synthetically curated purchase history
and modeled three ways — a purchase-cadence recommender, item-based collaborative filtering,
and a Generative Adversarial Network that synthesizes new shopping records.

## Authors

| Name | Email |
|---|---|
| Abhishek Ingle | abhingle@iu.edu |
| Bhargavi Jahagirdar | bjahagir@iu.edu |
| Naman Shah | shahnam@iu.edu |
| Navyasree Santhapet | nsantha@iu.edu |

Submitted May 2, 2024.

## Motivation

Retailers benefit from knowing not just *what* a customer buys but *when* they will need it
again. A digital twin — a continuously-updated model standing in for a real consumer — lets a
business simulate that behavior, personalize recommendations, and test retail strategies
without experimenting on real customers.

The central problem: no suitable public dataset of individual longitudinal grocery purchases
was available, so one was generated.

## The dataset

A synthetic purchase history for a **single consumer**, ~6,700 records, with fields
`Date`, `Item`, `Price`, `Quantity`, `Mode of Payment`, and `Coupon`.

The generator models three distinct purchase rhythms, which is what makes cadence prediction
meaningful:

| Category | Examples | Repurchase interval |
|---|---|---|
| Essentials | milk, eggs, bread | every 3–5 days |
| Frequently bought | — | every 10–15 days |
| Rarely bought | — | every 1–3 months |

**Feature engineering.** Because the data is synthetic there are no missing values or outliers.
Two derived features were added per row: `previous_purchase_date` and
`days_after_previous_purchase`. Categorical fields (`Item`, `Mode of Payment`) are label- or
one-hot encoded; `Price` and `Quantity` are standardized.

> **Note:** the CSVs the notebooks read (`grocery_data_mostrecent.csv`, `consumer_data.csv`)
> are not committed to this repository. You will need to supply or regenerate them.

## Repository contents

| File | Description |
|---|---|
| `DM_Recommendation_System.ipynb` | Cadence-based recommender. Computes the mean inter-purchase interval per item, projects each item's next purchase date from its last purchase, predicts quantity from the rounded historical mean, and writes a ranked `recommendations.csv` of what the consumer will need next. Written for Google Colab. |
| `Digitaltwin.ipynb` | Compact Keras GAN. A 32-dim latent generator (128 → 256 → 4, `tanh`) paired with a discriminator (256 → 128 → 1), sampling 100 synthetic purchase records and inverse-transforming `Price` and `Quantity` back to real units. |
| `DigitalTwin(GANmodel).ipynb` | The fuller GAN experiment. Builds per-customer purchase *sequences* (padded to fixed length), one-hot encodes items, and trains a regularized discriminator (512 → 256, LeakyReLU, dropout, L2) against the generator with exponential learning-rate decay after epoch 10. |
| `Digital_twin_report.txt` | Working writeup for the recommendation-system component: related work, methodology, results, references. |
| `DMFinalReport.pdf` | The submitted final report, covering this project alongside the team's rice crop identification project. |
| `Digitall_twin_output.png` | Screenshot of the recommender's precision/recall/F1 output. |
| `crop_identification_Sentinel 2 cloud filtering.ipynb` | Belongs to the rice crop project (see [RiceCropClassification](https://github.com/NavyasreeSR/RiceCropClassification)); included here because the final report spans both. |
| `File1.txt` | Scratch file from repository setup; not part of the project. |

## Approaches

### 1. Cadence-based recommendation

The simplest and most directly useful model. For each item, take the mean gap between
consecutive purchases, add it to the most recent purchase date, and rank the resulting
predicted dates. Quantity is the rounded historical average.

This produces output of the form:

```
1. Item: spaghetti sauce, Next Purchase Date: 2024-04-10, Predicted Quantity: 3
2. Item: tomatoes,        Next Purchase Date: 2024-04-11, Predicted Quantity: 3
3. Item: garlic powder,   Next Purchase Date: 2024-04-11, Predicted Quantity: 3
```

### 2. Item-based collaborative filtering (KNN Basic)

Item–item similarities computed with **cosine similarity**, then a KNN Basic model
(from the `surprise` library) identifies each item's *K* nearest neighbors and aggregates
them into recommendations. Chosen for being simple to interpret and tolerant of sparse data.
*K* was tuned on a train/test split.

### 3. Random Forest

A `RandomForestClassifier` tuned with `GridSearchCV` over `n_estimators` (100/200/300),
`max_depth`, `min_samples_split` (2/5/10), and `min_samples_leaf` (1/2/4).

### 4. Generative Adversarial Networks

GANs are used here to *augment* the dataset — generating synthetic operational data so the
digital twin has more behavior to learn from. The generator and discriminator play the usual
minimax game; the generator produces candidate purchase records and the discriminator judges
whether they look real.

## Results

**Recommendation quality** (single evaluation run):

```
Recommended Items: ['bread', 'rice', 'yogurt', 'mushrooms']
Actual Items:      ['bread', 'rice', 'mushrooms']
```

| Metric | Value |
|---|---|
| Precision | 0.75 |
| Recall | 0.67 |
| F1 Score | 0.56 |

**Random Forest:** accuracy of **0.36** on the digital twin task.

**GAN:** discriminator loss fell steadily across epochs while generator loss rose — the
discriminator outpaced the generator. Reported sample-quality metrics:

| Metric | Value |
|---|---|
| Inception Score (IS) | 6.149 |
| Fréchet Inception Distance (FID) | 5.814 |

The report is candid that these numbers leave room for improvement: the Random Forest and GAN
in particular proved difficult to tune, and the loss curves alone don't establish that the
generated samples are useful. Visual inspection and stronger evaluation were flagged as
needed follow-up.

## Running the notebooks

```bash
pip install pandas numpy scikit-learn tensorflow scikit-surprise matplotlib simpy
```

Supply the purchase-history CSV (`grocery_data_mostrecent.csv` for the recommender and
sequence GAN, `consumer_data.csv` for the compact GAN) in the working directory, then run
the notebooks in any order — they are independent.

`DM_Recommendation_System.ipynb` mounts Google Drive and reads from
`/content/drive/MyDrive/`; replace that path with a local one to run outside Colab.

## Known gaps

- The KNN Basic and Random Forest models are described in the report but the notebooks for
  them are not in this repository — only the cadence recommender and the two GANs are committed.
- The source CSVs and the data-generation script are not committed.
- The recommendation metrics come from a single evaluation on one consumer's history; they
  are indicative, not a rigorous benchmark.

## Future work

- Refine the models for higher accuracy, particularly the Random Forest and GAN.
- Extend to a **multi-consumer framework** to test scalability and surface broader market
  trends, rather than modeling a single shopper.
- Use GAN-generated behavior to simulate consumers under conditions not present in the
  observed data.

## References

1. Goodfellow, I. et al. *"Generative Adversarial Nets."* NeurIPS, 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
2. Radford, A.; Metz, L.; Chintala, S. *"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks."* 2015. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
3. Arjovsky, M.; Chintala, S.; Bottou, L. *"Wasserstein GAN."* 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
4. Garanayak, M.; Mohanty, S.; Jagadev, A.; Sahoo, S. *"Recommender System Using Item Based Collaborative Filtering (CF) and K-Means."* Int. J. Knowledge-based and Intelligent Engineering Systems, 23(1), 93–101, 2019.
5. Phorasim, P.; Yu, L. *"Movies Recommendation System Using Collaborative Filtering and K-Means."* Int. J. Advanced Computer Research, 7(29), 2017.
6. Gong, S. *"A Collaborative Filtering Recommendation Algorithm Based on User Clustering and Item Clustering."* Journal of Software, 5(7), 745–752, 2010.
7. TensorFlow, *"Generative Adversarial Network"* tutorial; PyTorch, *"DCGAN"* tutorial.

## Related

The final report also covers the team's rice crop identification work, documented separately
in [RiceCropClassification](https://github.com/NavyasreeSR/RiceCropClassification).

---

*Academic coursework submitted for B565 Data Mining, Indiana University Bloomington, Spring 2024.*
