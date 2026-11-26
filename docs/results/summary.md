# wmlab experiment results
Toy trigram LM on public-domain text, CPU only. z threshold 4 (one-sided p ~ 3.2e-5).

## E1 Detection vs length

| scheme | delta | gamma | tokens | n | tpr_at_z4 | mean_z | human_fpr_at_z4 |
|---|---|---|---|---|---|---|---|
| kgw | 1.0 | 0.25 | 16 | 100 | 0.02 | 1.494 | 0.0 |
| kgw | 1.0 | 0.25 | 32 | 100 | 0.09 | 2.287 | 0.0 |
| kgw | 1.0 | 0.25 | 64 | 100 | 0.3 | 3.425 | 0.0 |
| kgw | 1.0 | 0.25 | 128 | 100 | 0.75 | 4.661 | 0.0 |
| kgw | 1.0 | 0.25 | 200 | 100 | 0.96 | 5.632 | 0.0 |
| kgw | 2.0 | 0.25 | 16 | 100 | 0.44 | 3.758 | 0.0 |
| kgw | 2.0 | 0.25 | 32 | 100 | 0.89 | 5.137 | 0.0 |
| kgw | 2.0 | 0.25 | 64 | 100 | 1.0 | 7.096 | 0.0 |
| kgw | 2.0 | 0.25 | 128 | 100 | 1.0 | 10.023 | 0.0 |
| kgw | 2.0 | 0.25 | 200 | 100 | 1.0 | 12.538 | 0.0 |
| kgw | 3.0 | 0.25 | 16 | 100 | 0.86 | 4.996 | 0.0 |
| kgw | 3.0 | 0.25 | 32 | 100 | 1.0 | 7.196 | 0.0 |
| kgw | 3.0 | 0.25 | 64 | 100 | 1.0 | 10.243 | 0.0 |
| kgw | 3.0 | 0.25 | 128 | 100 | 1.0 | 14.197 | 0.0 |
| kgw | 3.0 | 0.25 | 200 | 100 | 1.0 | 17.432 | 0.0 |
| kgw | 4.0 | 0.25 | 16 | 100 | 0.98 | 5.855 | 0.0 |
| kgw | 4.0 | 0.25 | 32 | 100 | 0.98 | 8.314 | 0.0 |
| kgw | 4.0 | 0.25 | 64 | 100 | 0.99 | 11.683 | 0.0 |
| kgw | 4.0 | 0.25 | 128 | 100 | 1.0 | 16.449 | 0.0 |
| kgw | 4.0 | 0.25 | 200 | 100 | 1.0 | 20.324 | 0.0 |
| gumbel |  |  | 16 | 100 | 0.95 | 5.26 | 0.0 |
| gumbel |  |  | 32 | 100 | 1.0 | 7.916 | 0.0 |
| gumbel |  |  | 64 | 100 | 1.0 | 11.456 | 0.0 |
| gumbel |  |  | 128 | 100 | 1.0 | 16.302 | 0.0 |
| gumbel |  |  | 200 | 100 | 1.0 | 20.377 | 0.0 |

## E2 Robustness to edits

| scheme | attack | strength | n | tpr_at_z4 | mean_z |
|---|---|---|---|---|---|
| kgw d=2 | substitute | 0.0 | 100 | 1.0 | 12.432 |
| kgw d=2 | substitute | 0.1 | 100 | 1.0 | 10.365 |
| kgw d=2 | substitute | 0.2 | 100 | 1.0 | 8.525 |
| kgw d=2 | substitute | 0.3 | 100 | 1.0 | 6.952 |
| kgw d=2 | substitute | 0.5 | 100 | 0.46 | 3.922 |
| kgw d=2 | substitute | 0.7 | 100 | 0.03 | 1.842 |
| kgw d=2 | delete | 0.1 | 100 | 1.0 | 10.637 |
| kgw d=2 | insert | 0.1 | 100 | 1.0 | 10.743 |
| kgw d=2 | swap | 0.1 | 100 | 1.0 | 9.917 |
| kgw d=2 | delete | 0.2 | 100 | 1.0 | 8.84 |
| kgw d=2 | insert | 0.2 | 100 | 1.0 | 9.424 |
| kgw d=2 | swap | 0.2 | 100 | 1.0 | 8.007 |
| kgw d=2 | delete | 0.3 | 100 | 1.0 | 7.367 |
| kgw d=2 | insert | 0.3 | 100 | 1.0 | 8.261 |
| kgw d=2 | swap | 0.3 | 100 | 1.0 | 6.521 |
| kgw d=2 | paraphrase_toy | 0.3 | 100 | 1.0 | 10.966 |
| kgw d=2 | paraphrase_toy | 0.6 | 100 | 1.0 | 9.546 |
| kgw d=2 | paraphrase_toy | 1.0 | 100 | 1.0 | 7.743 |
| kgw d=2 | mix_into_human | 0.5 | 100 | 1.0 | 8.497 |
| kgw d=2 | mix_into_human | 0.25 | 100 | 0.94 | 5.708 |
| kgw d=2 | mix_into_human | 0.1 | 100 | 0.18 | 3.214 |
| kgw d=2 | zero_width | 0.5 | 100 | 0.0 | 1.013 |
| kgw d=2 | zero_width+normalise | 0.5 | 100 | 1.0 | 12.432 |
| gumbel | substitute | 0.0 | 100 | 1.0 | 20.347 |
| gumbel | substitute | 0.1 | 100 | 1.0 | 14.6 |
| gumbel | substitute | 0.2 | 100 | 1.0 | 9.914 |
| gumbel | substitute | 0.3 | 100 | 0.95 | 6.256 |
| gumbel | substitute | 0.5 | 100 | 0.05 | 2.224 |
| gumbel | substitute | 0.7 | 100 | 0.0 | 0.446 |
| gumbel | delete | 0.1 | 100 | 1.0 | 13.956 |
| gumbel | insert | 0.1 | 100 | 1.0 | 14.266 |
| gumbel | swap | 0.1 | 100 | 1.0 | 14.712 |
| gumbel | delete | 0.2 | 100 | 1.0 | 8.947 |
| gumbel | insert | 0.2 | 100 | 1.0 | 9.481 |
| gumbel | swap | 0.2 | 100 | 1.0 | 10.276 |
| gumbel | delete | 0.3 | 100 | 0.82 | 5.389 |
| gumbel | insert | 0.3 | 100 | 0.88 | 5.622 |
| gumbel | swap | 0.3 | 100 | 0.96 | 7.068 |
| gumbel | paraphrase_toy | 0.3 | 100 | 1.0 | 15.788 |
| gumbel | paraphrase_toy | 0.6 | 100 | 1.0 | 11.597 |
| gumbel | paraphrase_toy | 1.0 | 100 | 1.0 | 7.526 |
| gumbel | mix_into_human | 0.5 | 100 | 1.0 | 16.876 |
| gumbel | mix_into_human | 0.25 | 100 | 1.0 | 13.373 |
| gumbel | mix_into_human | 0.1 | 100 | 1.0 | 9.247 |
| gumbel | zero_width | 0.5 | 100 | 0.0 | 0.051 |
| gumbel | zero_width+normalise | 0.5 | 100 | 1.0 | 20.347 |

## E3 False positives on human text

| corpus | key | dedupe | repeat | threshold | n | empirical_fpr | theoretical_fpr |
|---|---|---|---|---|---|---|---|
| human | 0 | True | 1 | 2.0 | 200 | 0.01 | 2.28e-02 |
| human | 0 | True | 1 | 3.0 | 200 | 0.0 | 1.35e-03 |
| human | 0 | True | 1 | 4.0 | 200 | 0.0 | 3.17e-05 |
| human | 1 | True | 1 | 2.0 | 200 | 0.025 | 2.28e-02 |
| human | 1 | True | 1 | 3.0 | 200 | 0.005 | 1.35e-03 |
| human | 1 | True | 1 | 4.0 | 200 | 0.0 | 3.17e-05 |
| human | 2 | True | 1 | 2.0 | 200 | 0.03 | 2.28e-02 |
| human | 2 | True | 1 | 3.0 | 200 | 0.0 | 1.35e-03 |
| human | 2 | True | 1 | 4.0 | 200 | 0.0 | 3.17e-05 |
| human | 3 | True | 1 | 2.0 | 200 | 0.01 | 2.28e-02 |
| human | 3 | True | 1 | 3.0 | 200 | 0.0 | 1.35e-03 |
| human | 3 | True | 1 | 4.0 | 200 | 0.0 | 3.17e-05 |
| human | 4 | True | 1 | 2.0 | 200 | 0.015 | 2.28e-02 |
| human | 4 | True | 1 | 3.0 | 200 | 0.0 | 1.35e-03 |
| human | 4 | True | 1 | 4.0 | 200 | 0.0 | 3.17e-05 |
| alice | 0 | True | 1 | 2.0 | 47 | 0.0213 | 2.28e-02 |
| alice | 0 | True | 1 | 3.0 | 47 | 0.0 | 1.35e-03 |
| alice | 0 | True | 1 | 4.0 | 47 | 0.0 | 3.17e-05 |
| alice | 1 | True | 1 | 2.0 | 47 | 0.0 | 2.28e-02 |
| alice | 1 | True | 1 | 3.0 | 47 | 0.0 | 1.35e-03 |
| alice | 1 | True | 1 | 4.0 | 47 | 0.0 | 3.17e-05 |
| alice | 2 | True | 1 | 2.0 | 47 | 0.0 | 2.28e-02 |
| alice | 2 | True | 1 | 3.0 | 47 | 0.0 | 1.35e-03 |
| alice | 2 | True | 1 | 4.0 | 47 | 0.0 | 3.17e-05 |
| alice | 3 | True | 1 | 2.0 | 47 | 0.0 | 2.28e-02 |
| alice | 3 | True | 1 | 3.0 | 47 | 0.0 | 1.35e-03 |
| alice | 3 | True | 1 | 4.0 | 47 | 0.0 | 3.17e-05 |
| alice | 4 | True | 1 | 2.0 | 47 | 0.0 | 2.28e-02 |
| alice | 4 | True | 1 | 3.0 | 47 | 0.0 | 1.35e-03 |
| alice | 4 | True | 1 | 4.0 | 47 | 0.0 | 3.17e-05 |
| human x repeat | 0 | False | 1 | 2.0 | 200 | 0.04 | 2.28e-02 |
| human x repeat | 0 | False | 1 | 4.0 | 200 | 0.0 | 3.17e-05 |
| human x repeat | 0 | False | 3 | 2.0 | 200 | 0.09 | 2.28e-02 |
| human x repeat | 0 | False | 3 | 4.0 | 200 | 0.01 | 3.17e-05 |
| human x repeat | 0 | False | 10 | 2.0 | 200 | 0.195 | 2.28e-02 |
| human x repeat | 0 | False | 10 | 4.0 | 200 | 0.09 | 3.17e-05 |
| human x repeat | 0 | True | 1 | 2.0 | 200 | 0.02 | 2.28e-02 |
| human x repeat | 0 | True | 1 | 4.0 | 200 | 0.0 | 3.17e-05 |
| human x repeat | 0 | True | 3 | 2.0 | 200 | 0.02 | 2.28e-02 |
| human x repeat | 0 | True | 3 | 4.0 | 200 | 0.0 | 3.17e-05 |
| human x repeat | 0 | True | 10 | 2.0 | 200 | 0.02 | 2.28e-02 |
| human x repeat | 0 | True | 10 | 4.0 | 200 | 0.0 | 3.17e-05 |

## E4 Quality (perplexity) vs strength

| setting | n | median_ppl | mean_z | tpr_at_z4 |
|---|---|---|---|---|
| none | 100 | 16.94 | -0.524 | 0.0 |
| kgw d=0.5 | 100 | 17.62 | 2.212 | 0.02 |
| kgw d=1 | 100 | 18.92 | 5.184 | 0.89 |
| kgw d=2 | 100 | 23.68 | 10.844 | 1.0 |
| kgw d=4 | 100 | 40.21 | 17.844 | 1.0 |
| kgw d=8 | 100 | 53.74 | 19.89 | 1.0 |
| gumbel | 100 | 17.57 | 17.621 | 1.0 |

## E5 gamma and context width

| gamma | context_width | n | tokens | tpr_clean | mean_z_clean | tpr_sub20 | mean_z_sub20 |
|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 100 | 100 | 0.99 | 9.371 | 0.96 | 6.29 |
| 0.1 | 2 | 100 | 100 | 1.0 | 9.827 | 0.81 | 5.607 |
| 0.1 | 4 | 100 | 100 | 1.0 | 9.925 | 0.46 | 3.814 |
| 0.25 | 1 | 100 | 100 | 1.0 | 9.063 | 0.99 | 6.142 |
| 0.25 | 2 | 100 | 100 | 1.0 | 9.726 | 0.95 | 5.522 |
| 0.25 | 4 | 100 | 100 | 1.0 | 9.617 | 0.42 | 3.693 |
| 0.5 | 1 | 100 | 100 | 1.0 | 6.719 | 0.73 | 4.483 |
| 0.5 | 2 | 100 | 100 | 1.0 | 7.426 | 0.55 | 4.178 |
| 0.5 | 4 | 100 | 100 | 1.0 | 7.316 | 0.15 | 2.89 |

## E6 Image provenance bindings

| transform | n | hard_binding_valid | soft_binding_match | mean_dhash_distance |
|---|---|---|---|---|
| unchanged (metadata stripped) | 20 | 1.0 | 1.0 | 0.0 |
| png re-save | 20 | 1.0 | 1.0 | 0.0 |
| jpeg q95 | 20 | 0.0 | 1.0 | 0.1 |
| jpeg q70 | 20 | 0.0 | 1.0 | 0.35 |
| resize 50% | 20 | 0.0 | 1.0 | 0.05 |
| crop 10% | 20 | 0.0 | 0.35 | 11.75 |
| brightness +20% | 20 | 0.0 | 1.0 | 0.9 |
| different image | 20 | 0.0 | 0.0 | 29.15 |
