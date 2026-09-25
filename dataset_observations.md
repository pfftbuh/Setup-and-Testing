# Dataset Observations

## Overview

The dataset contains 186 session-level observations and 81 columns. Each row represents one recorded session. The dataset contains 79 potential predictor variables, a `session_id` identifier, and a binary `label`.

The predictor variables consist of:

- Eight gaze-distribution features: normalized gaze centroid, horizontal and vertical spread, elongation ratio, entropy, peak ratio, and coverage ratio.
- Sixty-four spatial heatmap features representing a grid of screen regions.
- Seven behavioral features: frantic eye-movement violations, forbidden-key violations, off-screen violations, duration violations, number of gaze transitions, percentage of time outside the center region, and overall violation rate.

## Class Distribution

The classes are approximately balanced:

- Label 0: 93 sessions, or 50.0% of the dataset.
- Label 1: 93 sessions, or 50.0% of the dataset.

This balance is useful for binary classification because the model is less likely to achieve an apparently high accuracy simply by predicting the majority class.

## Data Quality

No missing values were observed in the dataset. All predictor columns contain numeric values, while `session_id` is a string identifier and `label` is the target variable. No duplicate session IDs were found in the inspected data.

## Captured Session Video Clips Analysis

Each session logs automated `.mp4` video clips whenever a behavioral violation is triggered (e.g., frantic eye movements, forbidden key shortcuts, eyes off-screen, or prolonged directional looks). Across all 186 sessions in the dataset, a total of **3,135 video clips** were captured.

### Overall & Class-Level Clip Counts

| Metric | Cheating (Label 1) | Non-Cheating (Label 0) | Overall Dataset / Difference |
|---|---:|---:|---:|
| **Total Sessions** | 93 | 93 | 186 sessions |
| **Total Captured Clips** | 2,035 | 1,100 | 3,135 clips |
| **Mean Clips / Session** | **21.88** | **11.83** | **+10.05 clips (+84.9%)** |
| **Median Clips / Session** | 20.0 | 11.0 | +9.0 clips |
| **Standard Deviation** | 9.29 | 6.29 | — |
| **Min / Max Range** | 4 to 57 | 1 to 38 | — |

### Key Observations & Comparison

- **Substantial Increase in Captured Evidence**: Cheating sessions captured an overall average of **21.88 clips per session**, compared to **11.83 clips per session** for non-cheating sessions. This represents an average difference of **+10.05 clips per session** (an 84.9% increase in recorded video evidence).
- **Statistical Significance**: A two-sample t-test ($t = 8.44, p = 2.67 \times 10^{-15}$) and Mann-Whitney U test ($U = 7,370.5, p = 1.76 \times 10^{-18}$) confirm that cheating sessions produce significantly more violation clips than non-cheating sessions.
- **Violation Category Breakdown**:
  - **Forbidden-Key Clips**: Cheating sessions averaged **5.47 clips/session** (509 total), whereas non-cheating sessions averaged **0.03 clips/session** (3 total).
  - **Frantic Eye Movement Clips**: Cheating sessions averaged **10.67 clips/session** (992 total) vs. **8.26 clips/session** (768 total) in non-cheating sessions.
  - **Off-Screen Violations**: Cheating sessions averaged **1.42 clips/session** (132 total) vs. **0.41 clips/session** (38 total) in non-cheating sessions.
  - **Duration Gaze Violations**: Cheating sessions averaged **4.32 clips/session** (402 total) vs. **3.13 clips/session** (291 total) in non-cheating sessions.

This difference aligns with expectations, as cheating sessions trigger considerably more hotkey violations, off-screen gaze diversions, and frantic eye patterns, directly resulting in a higher volume of auto-recorded video evidence clips.

### Violation Log Breakdown by Type

The table below aggregates every individual violation row logged in the per-session `session_log_*.csv` files across all sessions in the `sessions/cheating` and `sessions/non_cheating` folders, grouped by violation label and broader violation category, and split by class.

| Violation Label | Group | Cheating | Non-Cheating | Total |
|---|---|---:|---:|---:|
| eyes_off_screen | Off-screen | 41 | 15 | 56 |
| face_off_screen | Off-screen | 95 | 26 | 121 |
| Up-Left_duration | Sustained Direction Duration | 68 | 24 | 92 |
| Up-Center_duration | Sustained Direction Duration | 33 | 118 | 151 |
| Down-Center_duration | Sustained Direction Duration | 155 | 130 | 285 |
| Up-Right_duration | Sustained Direction Duration | 18 | 1 | 19 |
| Center-Left_duration | Sustained Direction Duration | 80 | 2 | 82 |
| Down-Right_duration | Sustained Direction Duration | 26 | 5 | 31 |
| Center-Right_duration | Sustained Direction Duration | 13 | 12 | 25 |
| Down-Left_duration | Sustained Direction Duration | 14 | 1 | 15 |
| frantic_eye_movement | Frantic Movement | 1,016 | 795 | 1,811 |
| forbidden_key_ctrl_c | Forbidden Key | 351 | 0 | 351 |
| forbidden_key_ctrl_v | Forbidden Key | 142 | 0 | 142 |
| forbidden_key_windows | Forbidden Key | 6 | 0 | 6 |
| forbidden_key_alt_tab | Forbidden Key | 11 | 2 | 13 |
| forbidden_key_f11 | Forbidden Key | 0 | 1 | 1 |
| **Total** | | **2,069** | **1,132** | **3,201** |

### Per-Session Log Summary

The table below lists every session folder under `sessions/cheating` and `sessions/non_cheating`, derived directly from each session's `session_log_*.csv` file. Log Rows is the total number of logged gaze intervals, Flagged Events is the count of rows whose `Violation label` is not `normal`, and Evidence Clips is the number of `.mp4` files captured in that session's folder.

| Session # | Category | Folder | Log Timestamp | Log Rows | Flagged Events | Evidence Clips |
|---|---|---|---|---:|---:|---:|
| 1 | Cheating | session_01705ccfc6e84812a9f13609d0f678cb | 2026-08-26 10:28:03 | 359 | 19 | 19 |
| 2 | Cheating | session_05de184384864e618ef35e1edba28aaf | 2026-08-27 15:31:46 | 256 | 14 | 14 |
| 3 | Cheating | session_09d4368e78d348fe9b0622e69d6cdd5d | 2026-08-25 16:35:44 | 1281 | 45 | 44 |
| 4 | Cheating | session_0cd911a71c2a481bae94fd688d8fc5eb | 2026-08-25 09:50:41 | 979 | 57 | 57 |
| 5 | Cheating | session_100db0f6cdf54ef4baa19c7e9282a119 | 2026-08-26 11:39:52 | 458 | 23 | 23 |
| 6 | Cheating | session_10b60786199e448ea464341e68ef74f1 | 2026-08-27 11:36:55 | 551 | 22 | 20 |
| 7 | Cheating | session_12d1e33fda9f4429962b4c59c9c6bd4a | 2026-08-27 12:55:26 | 549 | 23 | 23 |
| 8 | Cheating | session_147ca42112b547ee9cc1b1a63507c519 | 2026-08-28 11:12:31 | 510 | 18 | 18 |
| 9 | Cheating | session_1860988281f14c67bc06d7e91ac4da30 | 2026-08-28 10:04:00 | 346 | 22 | 22 |
| 10 | Cheating | session_19f2d53c700f480ebc267c3ea89cbc33 | 2026-08-27 09:41:45 | 1547 | 35 | 35 |
| 11 | Cheating | session_248e91193f904874ab3382373192600f | 2026-08-24 12:46:19 | 484 | 20 | 19 |
| 12 | Cheating | session_33255b3cd8fd41c38307da1ffa18cd58 | 2026-08-27 13:55:48 | 294 | 27 | 27 |
| 13 | Cheating | session_361139da56094c289aaca4d036fafd9f | 2026-08-27 09:16:37 | 1008 | 41 | 41 |
| 14 | Cheating | session_38e43a19a2284b5884d01266ffe5bcd2 | 2026-08-27 10:48:56 | 450 | 11 | 11 |
| 15 | Cheating | session_40e6d5311a604fc19b6b2a0fc08e66e2 | 2026-08-27 09:31:01 | 427 | 33 | 32 |
| 16 | Cheating | session_46a435b86f514519b515bce0473889f1 | 2026-08-27 15:56:36 | 566 | 23 | 23 |
| 17 | Cheating | session_47c95126f430432393edf60214943044 | 2026-08-28 10:20:53 | 428 | 20 | 20 |
| 18 | Cheating | session_491449cba73f401dac98e6cf2d260687 | 2026-08-25 08:16:48 | 343 | 20 | 19 |
| 19 | Cheating | session_4992ab81e99f4a0895d84fc830afd3f7 | 2026-08-24 13:00:49 | 436 | 17 | 17 |
| 20 | Cheating | session_4cd44ea62cc4449e879b4a7ce3bade08 | 2026-08-26 13:50:57 | 297 | 15 | 15 |
| 21 | Cheating | session_4e1eb97dfcdc4a4b83fb4b9c9e392306 | 2026-08-27 13:29:56 | 395 | 19 | 19 |
| 22 | Cheating | session_4ea379d31391499cbc7a510267860239 | 2026-08-25 11:10:51 | 481 | 42 | 42 |
| 23 | Cheating | session_51faeb57d0ec41a18aa81cf2a7174e76 | 2026-08-26 10:15:29 | 582 | 26 | 26 |
| 24 | Cheating | session_554f767672234d57b345f7a352e63fe3 | 2026-08-27 15:49:14 | 291 | 13 | 13 |
| 25 | Cheating | session_59b4e2a3797b48bf9932c30a39238146 | 2026-08-26 10:37:45 | 389 | 17 | 16 |
| 26 | Cheating | session_5e0e9408fd8f462f90b9ba6ea9760167 | 2026-08-27 11:14:46 | 138 | 5 | 4 |
| 27 | Cheating | session_60f09c2ac1054c7aa9550238b11bf6e2 | 2026-08-27 13:23:07 | 473 | 22 | 21 |
| 28 | Cheating | session_614853a387114f23818f0bd826f9529d | 2026-08-28 09:26:24 | 310 | 12 | 12 |
| 29 | Cheating | session_62b4f37f9566455c8e40f66aa8057533 | 2026-08-27 11:57:41 | 485 | 17 | 17 |
| 30 | Cheating | session_62e6f7cd43424d3bbcccdb6b8c6742c6 | 2026-08-26 13:01:54 | 385 | 16 | 16 |
| 31 | Cheating | session_64ff7c6042be4b08b1eaa1fcb28bd621 | 2026-08-25 16:11:10 | 590 | 18 | 18 |
| 32 | Cheating | session_68b15e05a5ae4daca9bdc9a5ed527334 | 2026-08-27 12:41:22 | 688 | 13 | 13 |
| 33 | Cheating | session_6aed13072a424c79b9b68e8787d245ba | 2026-08-27 13:14:45 | 336 | 19 | 19 |
| 34 | Cheating | session_741ab24caf9e4d3fad2d6818b3003c73 | 2026-08-27 15:13:47 | 495 | 20 | 19 |
| 35 | Cheating | session_74ca31a0192c4804828d7e7ba9694638 | 2026-08-26 11:25:01 | 305 | 18 | 18 |
| 36 | Cheating | session_762199fde1714a66a4d573ca708ea95b | 2026-08-27 14:50:10 | 978 | 47 | 47 |
| 37 | Cheating | session_78022aff4fa145fdac5261a0d0018ccf | 2026-08-27 08:44:17 | 297 | 25 | 24 |
| 38 | Cheating | session_79c419ef5ef148368bde611bfdc35de1 | 2026-08-27 12:09:47 | 531 | 20 | 20 |
| 39 | Cheating | session_79e01b5bc77144b2935d94c47bce37d3 | 2026-08-25 12:42:19 | 556 | 30 | 30 |
| 40 | Cheating | session_7b82dce22ffe4e588f52259e7a4b37b8 | 2026-08-28 11:03:00 | 442 | 18 | 17 |
| 41 | Cheating | session_815ebb529ae64f678fdcaa67e7e2bc51 | 2026-08-28 08:12:34 | 433 | 22 | 22 |
| 42 | Cheating | session_83bc2309a0a54b7aa7138eed550c0ce1 | 2026-08-26 09:40:00 | 454 | 18 | 17 |
| 43 | Cheating | session_844c41b1697e4f55981a48185e8c1e1a | 2026-08-26 12:55:19 | 354 | 14 | 14 |
| 44 | Cheating | session_85334f32d1f64930bff5979e9ee3bcdd | 2026-08-27 09:58:32 | 511 | 39 | 39 |
| 45 | Cheating | session_85ed145d863741e4b68f7fb466ad46c6 | 2026-08-28 08:25:47 | 522 | 27 | 27 |
| 46 | Cheating | session_87a82c398633482cb334f4d033e666e1 | 2026-08-24 13:10:39 | 469 | 19 | 18 |
| 47 | Cheating | session_88c7d95c6f8e4e7bb6df45318dbdea12 | 2026-08-26 13:35:59 | 520 | 27 | 26 |
| 48 | Cheating | session_89dd28a92f9a4670ac38cd1bdd30d248 | 2026-08-27 11:06:25 | 389 | 17 | 16 |
| 49 | Cheating | session_8bfbf9b8ff344d05a0ce41082bccdd33 | 2026-08-27 13:06:59 | 456 | 12 | 12 |
| 50 | Cheating | session_8c4ffffa44a1467482119f9baa510775 | 2026-08-26 13:23:07 | 539 | 17 | 16 |
| 51 | Cheating | session_8e259f318ffe41d18323dc1ac179af78 | 2026-08-27 10:19:38 | 240 | 23 | 23 |
| 52 | Cheating | session_920d2eb010394d32b500dffff0cf770b | 2026-08-27 15:03:22 | 524 | 21 | 21 |
| 53 | Cheating | session_93bcea53dbba4ec4944f6fde950b7402 | 2026-08-26 11:54:17 | 217 | 10 | 9 |
| 54 | Cheating | session_95435cdcd5b847e49c5862b00c0ad1c9 | 2026-08-27 11:26:04 | 356 | 28 | 27 |
| 55 | Cheating | session_956d95b662c84fd38a931ff31f9934e0 | 2026-08-24 13:21:14 | 1032 | 41 | 41 |
| 56 | Cheating | session_980d3c56c0fb45b78ff48c63f1c3ad0c | 2026-08-24 11:42:12 | 451 | 22 | 22 |
| 57 | Cheating | session_99824d4fcaa84dcf91d743b8262ead63 | 2026-08-26 09:26:47 | 477 | 20 | 20 |
| 58 | Cheating | session_9affad06246e4c8b8c80c8af6fedef5d | 2026-08-26 08:37:57 | 361 | 16 | 16 |
| 59 | Cheating | session_a0aed977e06e4d02889831920ee3a364 | 2026-08-26 14:45:37 | 479 | 22 | 22 |
| 60 | Cheating | session_a3191f6eb01a4042abd63da8edc5b6af | 2026-08-24 13:36:06 | 423 | 29 | 29 |
| 61 | Cheating | session_a537888910834ab98987d517c797cfe4 | 2026-08-26 15:05:11 | 504 | 26 | 25 |
| 62 | Cheating | session_a786863df05d4a28b727837d5e142a90 | 2026-08-27 14:06:04 | 103 | 15 | 14 |
| 63 | Cheating | session_aa170149eb8c473693a5fd1c40609634 | 2026-08-28 10:12:04 | 422 | 21 | 21 |
| 64 | Cheating | session_abf7f2b46a404e43bb07a037544a44bf | 2026-08-27 15:38:26 | 225 | 13 | 13 |
| 65 | Cheating | session_b2aaf02d12ed4689b06147357e085e26 | 2026-08-24 12:17:52 | 593 | 25 | 24 |
| 66 | Cheating | session_b415554bdb004975a14f4b4f60e8c402 | 2026-08-27 11:44:51 | 692 | 34 | 33 |
| 67 | Cheating | session_b45e66bdac564d15b36102182255ba66 | 2026-08-27 13:37:38 | 373 | 20 | 20 |
| 68 | Cheating | session_b48a01f38b7544fc833358cce2769b52 | 2026-08-27 10:57:43 | 407 | 20 | 20 |
| 69 | Cheating | session_b4b56d4c2bf44065bab2029f0e2cad44 | 2026-08-24 12:06:31 | 463 | 27 | 26 |
| 70 | Cheating | session_b5231ae90a2643f5bc20c7e544366111 | 2026-08-26 14:56:07 | 366 | 17 | 16 |
| 71 | Cheating | session_b80b18e1ac0848909db3a513e02839bf | 2026-08-26 08:48:18 | 283 | 14 | 14 |
| 72 | Cheating | session_c43f17ea14554657aa7479a7182ac0e7 | 2026-08-26 11:07:35 | 580 | 24 | 23 |
| 73 | Cheating | session_c633edc8afee42b6b63aabc6585477d5 | 2026-08-27 11:17:42 | 369 | 12 | 11 |
| 74 | Cheating | session_c7afaf7caec5457188057d2c296c91e3 | 2026-08-24 16:25:40 | 683 | 27 | 27 |
| 75 | Cheating | session_c9fe1895ac6f4bbc97e7faa4dfd0fe4b | 2026-08-25 08:27:04 | 695 | 26 | 26 |
| 76 | Cheating | session_ceafb8b8801c47e2b8f93216e5d42349 | 2026-08-27 13:45:08 | 281 | 17 | 17 |
| 77 | Cheating | session_d41c589fab2b4f0694464fa1d96c1bb6 | 2026-08-27 10:33:32 | 819 | 35 | 35 |
| 78 | Cheating | session_d5b12b929cbf4287bb982ff203cc81a9 | 2026-08-26 10:57:49 | 348 | 20 | 20 |
| 79 | Cheating | session_d7b0aaa7c51d475cb0ce12af934793c0 | 2026-08-27 10:10:13 | 372 | 15 | 15 |
| 80 | Cheating | session_d81acb3d133f42bab42f0895ccbeecf2 | 2026-08-26 12:45:49 | 367 | 8 | 8 |
| 81 | Cheating | session_dacff460d3c44787aae4302601fe1a15 | 2026-08-25 12:25:06 | 891 | 45 | 44 |
| 82 | Cheating | session_dcb2051fad5a4297ae5b057f91b54b9c | 2026-08-26 11:16:42 | 267 | 18 | 17 |
| 83 | Cheating | session_dd3b8c9d70e74e34b0e91ffe70919971 | 2026-08-28 09:48:27 | 852 | 41 | 40 |
| 84 | Cheating | session_e47fbbd30ed54a7d989257d609873855 | 2026-08-24 16:37:54 | 376 | 22 | 22 |
| 85 | Cheating | session_e649e4dfafdf4ac3a8d6baae28f0d4fc | 2026-08-25 16:55:34 | 578 | 22 | 22 |
| 86 | Cheating | session_eada1f2893c2465684eab955c1343928 | 2026-08-26 11:46:55 | 427 | 20 | 19 |
| 87 | Cheating | session_ed423862c86a49c99fa24f4e6ff93877 | 2026-08-24 11:57:27 | 333 | 20 | 20 |
| 88 | Cheating | session_eeabb109b1ea4cae82a2ab66b3f2026d | 2026-08-26 10:48:36 | 382 | 18 | 17 |
| 89 | Cheating | session_f3fe34d6a7574b918d9b892ff7052de4 | 2026-08-26 09:12:33 | 375 | 13 | 13 |
| 90 | Cheating | session_f626002ef2f74b519401317d98f48829 | 2026-08-25 16:23:53 | 463 | 23 | 22 |
| 91 | Cheating | session_f63c09bfe1eb47a68aaed7c6224e627a | 2026-08-26 11:31:27 | 237 | 14 | 13 |
| 92 | Cheating | session_f7e9b8ca8b3d4b41823bbdb70cb2a788 | 2026-08-26 09:57:29 | 394 | 17 | 17 |
| 93 | Cheating | session_ff05b5ce81574447b32f1f81507f1d02 | 2026-08-27 15:21:31 | 269 | 14 | 14 |
| 94 | Non-Cheating | session_02f6647099be482aa41b1bfe5bdb0bd5 | 2026-08-26 10:12:02 | 193 | 10 | 9 |
| 95 | Non-Cheating | session_09510fcef51341d89bc860d3d4a49c02 | 2026-08-26 15:01:46 | 267 | 11 | 11 |
| 96 | Non-Cheating | session_0b757f1e1d864431b2833fca706d3de1 | 2026-08-27 12:52:30 | 116 | 9 | 8 |
| 97 | Non-Cheating | session_0f283c6e8a2a41c88531876dbccaf66a | 2026-08-28 10:17:59 | 92 | 6 | 5 |
| 98 | Non-Cheating | session_141ba797f345475984edebb334bb94d3 | 2026-08-26 11:44:19 | 102 | 6 | 6 |
| 99 | Non-Cheating | session_16f7081aa78141cdbc4ac5c2a6e56617 | 2026-08-27 11:23:17 | 172 | 9 | 8 |
| 100 | Non-Cheating | session_1fe3688ab3074e09a9bd9901da5cc77e | 2026-08-24 12:56:22 | 286 | 17 | 17 |
| 101 | Non-Cheating | session_2047213c712741088a17f93a741c629c | 2026-08-26 12:52:14 | 207 | 12 | 11 |
| 102 | Non-Cheating | session_20899a71736b49b79b8fac1212357bc3 | 2026-08-28 11:09:28 | 244 | 9 | 9 |
| 103 | Non-Cheating | session_211cc9e12d30427bb99b5d409fb47e90 | 2026-08-26 11:50:44 | 196 | 8 | 8 |
| 104 | Non-Cheating | session_227e9e09d442499abb4e46af09940436 | 2026-08-26 08:45:22 | 149 | 7 | 7 |
| 105 | Non-Cheating | session_290b2a3f63734328852729316c5d60a8 | 2026-08-27 11:53:54 | 510 | 15 | 15 |
| 106 | Non-Cheating | session_2fba67581ca5472487c1e2622428a1fd | 2026-08-25 16:51:12 | 247 | 12 | 11 |
| 107 | Non-Cheating | session_3294ed22ce1a420f82261ff60f521414 | 2026-08-25 08:23:35 | 184 | 11 | 10 |
| 108 | Non-Cheating | session_39b3b152fd204671990cfdced37194ee | 2026-08-24 11:37:02 | 198 | 14 | 13 |
| 109 | Non-Cheating | session_3d148f9bf3d847139320307666cc80f0 | 2026-08-27 11:33:57 | 200 | 13 | 12 |
| 110 | Non-Cheating | session_3dd82ffdb64643f7a9f9edb9390d6df7 | 2026-08-28 09:23:38 | 176 | 6 | 6 |
| 111 | Non-Cheating | session_3ea645740c864b43b576658338e02187 | 2026-08-27 09:56:06 | 189 | 8 | 8 |
| 112 | Non-Cheating | session_3f9c79b70b1e4eb6bb7dedb4261b238f | 2026-08-27 13:12:28 | 109 | 8 | 8 |
| 113 | Non-Cheating | session_41e84e5483e44d3ba7a81db8f03038a2 | 2026-08-24 12:39:16 | 444 | 27 | 27 |
| 114 | Non-Cheating | session_425542a2ac854c9f884472d86a4dbe21 | 2026-08-26 11:13:30 | 143 | 12 | 12 |
| 115 | Non-Cheating | session_4384fc822c6b4278ae349391edbeb1d6 | 2026-08-27 15:00:34 | 207 | 10 | 9 |
| 116 | Non-Cheating | session_4420f4e4499d42148c9b2cf5ed1a97ce | 2026-08-26 09:21:32 | 454 | 30 | 30 |
| 117 | Non-Cheating | session_4667686ed48a4de08210cf068f4f49b9 | 2026-08-26 10:24:37 | 203 | 13 | 13 |
| 118 | Non-Cheating | session_4b30383221d740a9ad8e03a5c84b3321 | 2026-08-26 13:32:14 | 230 | 11 | 11 |
| 119 | Non-Cheating | session_4cd169afe5b446f6a13d56db44946027 | 2026-08-27 14:02:25 | 112 | 16 | 16 |
| 120 | Non-Cheating | session_5003614b2be044ec9c6d0800f863f337 | 2026-08-26 12:59:18 | 77 | 13 | 13 |
| 121 | Non-Cheating | session_5070a5c3712a409e9130229f26ececc8 | 2026-08-24 13:16:40 | 335 | 19 | 19 |
| 122 | Non-Cheating | session_531fd0d9fd624c88813609b4fdec247c | 2026-08-28 08:21:55 | 221 | 12 | 12 |
| 123 | Non-Cheating | session_5330e3e4aea04b26bb70a0ad5681f6b5 | 2026-08-24 13:32:31 | 233 | 10 | 10 |
| 124 | Non-Cheating | session_53bfc079e9454eb8b4a6dca842130f0b | 2026-08-26 13:14:42 | 701 | 39 | 38 |
| 125 | Non-Cheating | session_56da31a494eb40e3b47344da6e05a8a9 | 2026-08-27 14:47:05 | 125 | 7 | 6 |
| 126 | Non-Cheating | session_591d01cb80994e97ac0792c844c31129 | 2026-08-27 09:37:36 | 206 | 11 | 11 |
| 127 | Non-Cheating | session_5c23f02c089049c0ab021824fbf8b0ef | 2026-08-27 10:16:14 | 113 | 9 | 9 |
| 128 | Non-Cheating | session_5c7df5a171b34c96b4fb4bf403d83fd6 | 2026-08-28 11:00:00 | 257 | 13 | 13 |
| 129 | Non-Cheating | session_5ccb7c7f5a1145d48c8e8ef583f41699 | 2026-08-24 11:53:42 | 113 | 9 | 9 |
| 130 | Non-Cheating | session_5fc2cdde8baa479e99b01b07609646f2 | 2026-08-27 10:06:57 | 220 | 10 | 10 |
| 131 | Non-Cheating | session_607e7f830f1e4e3cb31aae595e00dba4 | 2026-08-27 13:27:27 | 89 | 5 | 5 |
| 132 | Non-Cheating | session_64cb355f03bf4b508606d10d07adbdf4 | 2026-08-25 08:12:34 | 122 | 8 | 8 |
| 133 | Non-Cheating | session_65a6e1a50bc04fa1bc968df0f2ca742e | 2026-08-26 11:03:45 | 353 | 13 | 13 |
| 134 | Non-Cheating | session_6dad1061e4a247a9ba9fc25d414d2e80 | 2026-08-27 15:28:31 | 164 | 7 | 7 |
| 135 | Non-Cheating | session_70f3c13f8b214ab9b7a270abd93bbc4a | 2026-08-26 14:52:27 | 324 | 13 | 12 |
| 136 | Non-Cheating | session_733fea50b4a84631abd7c47bb083d552 | 2026-08-26 09:09:13 | 202 | 7 | 6 |
| 137 | Non-Cheating | session_73860cabfdc04dee93082afac209d4d9 | 2026-08-27 10:29:54 | 216 | 12 | 12 |
| 138 | Non-Cheating | session_76c9cf0e57b84eb489804a85a2759687 | 2026-08-26 09:36:06 | 297 | 15 | 15 |
| 139 | Non-Cheating | session_7f56ff2619794811a358daff63450ed2 | 2026-08-27 12:37:43 | 334 | 17 | 16 |
| 140 | Non-Cheating | session_813234a27e504dc3b55f6f6e8bcc0561 | 2026-08-28 10:01:35 | 112 | 8 | 7 |
| 141 | Non-Cheating | session_8239ee853ef64d808ffe51eab713ce38 | 2026-08-27 13:02:50 | 207 | 16 | 15 |
| 142 | Non-Cheating | session_840da75e8fd24d2ba65113dcfaf29b22 | 2026-08-24 16:20:37 | 480 | 14 | 13 |
| 143 | Non-Cheating | session_8546e20627594735b45468c34e8a41cc | 2026-08-27 10:44:10 | 473 | 23 | 23 |
| 144 | Non-Cheating | session_8705882700ca4baca7ec4676bc50ccd2 | 2026-08-26 09:51:41 | 439 | 22 | 22 |
| 145 | Non-Cheating | session_94ab41d83d844d7691bdbe1ca96bad90 | 2026-08-26 11:22:22 | 109 | 8 | 7 |
| 146 | Non-Cheating | session_9631943b5ce0432097298d6d72977aa8 | 2026-08-27 15:53:02 | 280 | 16 | 16 |
| 147 | Non-Cheating | session_9ade1e2f731347b685f6183c1f2e6734 | 2026-08-27 11:02:54 | 205 | 12 | 12 |
| 148 | Non-Cheating | session_9d0469e83e1541a181a205cc3539064f | 2026-08-25 16:03:20 | 607 | 38 | 38 |
| 149 | Non-Cheating | session_9ea7aad7a8cd495fafd655ba4c0b0ed7 | 2026-08-26 10:43:40 | 208 | 10 | 10 |
| 150 | Non-Cheating | session_a0d2c8ec5dbe46e993574795573def31 | 2026-08-24 13:07:24 | 266 | 13 | 13 |
| 151 | Non-Cheating | session_a2fe402f8cc14ea1a188699453cb3d9d | 2026-08-27 15:45:30 | 117 | 7 | 7 |
| 152 | Non-Cheating | session_aadac2e407894877b85164efc620b06d | 2026-08-27 08:40:23 | 251 | 12 | 12 |
| 153 | Non-Cheating | session_aadb126b94244cf1b6f57e898f525c0f | 2026-08-27 15:10:33 | 254 | 13 | 12 |
| 154 | Non-Cheating | session_aed6641602be4138bed7c563a8d8faed | 2026-08-27 13:35:02 | 136 | 8 | 8 |
| 155 | Non-Cheating | session_b1086f4c3b4440a19f46f3cd2e3e5189 | 2026-08-26 10:34:40 | 159 | 9 | 9 |
| 156 | Non-Cheating | session_b1a687bc6a2b407f92fd108f940f9a56 | 2026-08-27 15:18:45 | 77 | 4 | 3 |
| 157 | Non-Cheating | session_b34fe886b68447a28543d5d8e10100a7 | 2026-08-26 11:29:31 | 79 | 6 | 6 |
| 158 | Non-Cheating | session_b3d7cf58280f443187fd8a92a0555097 | 2026-08-27 10:54:51 | 311 | 5 | 4 |
| 159 | Non-Cheating | session_b5b0349454fe44ba9653655f9878d354 | 2026-08-27 15:35:18 | 99 | 6 | 6 |
| 160 | Non-Cheating | session_b6c373dcfe294b44b887bd763a0fbb20 | 2026-08-27 09:27:59 | 200 | 12 | 11 |
| 161 | Non-Cheating | session_b812d0984b54419c83de96902ec8ecc1 | 2026-08-27 12:05:43 | 200 | 15 | 14 |
| 162 | Non-Cheating | session_bb33aa5997e643a9894115710253af75 | 2026-08-26 13:31:13 | 30 | 2 | 1 |
| 163 | Non-Cheating | session_bd5046f461f94604bc2bae627202bc76 | 2026-08-26 08:33:56 | 219 | 16 | 16 |
| 164 | Non-Cheating | session_bf4cb9ffc52d4f88a1cf067ce1ea8307 | 2026-08-27 13:42:34 | 172 | 8 | 7 |
| 165 | Non-Cheating | session_c25a4adf911145b79f8e5ee8631c2466 | 2026-08-27 13:51:42 | 326 | 16 | 16 |
| 166 | Non-Cheating | session_c6cee29ab9374e418cf2c89faef5e484 | 2026-08-25 12:19:41 | 209 | 14 | 14 |
| 167 | Non-Cheating | session_c87969d85fdf43f58099db91d34f0c40 | 2026-08-25 16:19:35 | 167 | 9 | 9 |
| 168 | Non-Cheating | session_c93a1622a59849b6b0bf3c9cdcf85e2e | 2026-08-26 11:35:50 | 248 | 10 | 9 |
| 169 | Non-Cheating | session_c9b8d1261d7a4bb8a9199e66977030e3 | 2026-08-25 16:31:53 | 201 | 11 | 11 |
| 170 | Non-Cheating | session_cdbb74f9573f467a8ed7c088e9898827 | 2026-08-28 10:09:04 | 100 | 4 | 4 |
| 171 | Non-Cheating | session_cde1ee5437ef43d4bb7aa7fd89049fcb | 2026-08-27 11:41:45 | 250 | 11 | 11 |
| 172 | Non-Cheating | session_cfbe8220f18d46e4aeb444f491b588b6 | 2026-08-27 09:12:37 | 172 | 11 | 11 |
| 173 | Non-Cheating | session_d33097fb7b42410690798a162a3ec85b | 2026-08-27 11:11:23 | 252 | 15 | 15 |
| 174 | Non-Cheating | session_d433bba853b441e485b2482f68b68e34 | 2026-08-26 10:54:29 | 145 | 9 | 9 |
| 175 | Non-Cheating | session_d95b6f10d73c4a8b948c4247300a86df | 2026-08-28 08:08:29 | 191 | 11 | 11 |
| 176 | Non-Cheating | session_dcbcf9d39752489fa2c781a308a85d1d | 2026-08-25 12:38:13 | 196 | 13 | 13 |
| 177 | Non-Cheating | session_dcc2b94ecb4947d0ad0a1da1cb0acd77 | 2026-08-25 11:06:24 | 187 | 13 | 12 |
| 178 | Non-Cheating | session_e522b30610d1465d8666e9ab1444591b | 2026-08-26 14:43:03 | 202 | 10 | 10 |
| 179 | Non-Cheating | session_e786057e26af4fdaa16d703e6dccd472 | 2026-08-25 09:44:59 | 168 | 27 | 27 |
| 180 | Non-Cheating | session_e9334496e16445d8b6844224b899a2dc | 2026-08-24 12:13:44 | 261 | 14 | 14 |
| 181 | Non-Cheating | session_f1bc6c823601424fb4d46adf129e0422 | 2026-08-24 16:34:28 | 179 | 9 | 8 |
| 182 | Non-Cheating | session_f6ad6fadf6d54ee09e37a0703505b3bf | 2026-08-24 12:03:21 | 142 | 9 | 9 |
| 183 | Non-Cheating | session_f9c0e59dbc81486b8040947ebd59e7b3 | 2026-08-27 13:19:38 | 249 | 10 | 10 |
| 184 | Non-Cheating | session_fa00e8251e764c728461586a5270d9af | 2026-08-26 12:41:41 | 147 | 10 | 9 |
| 185 | Non-Cheating | session_fcc314a8cfcd40b3ae0ebc9db0b0048e | 2026-08-26 13:46:32 | 238 | 16 | 15 |
| 186 | Non-Cheating | session_fe6cbad26e8f4358966977df2beb8c70 | 2026-08-28 09:43:59 | 279 | 18 | 17 |

## Observed Feature Patterns

Sessions assigned to label 1 generally show more active and dispersed gaze behavior than sessions assigned to label 0. The strongest observed differences include:

- A higher number of gaze transitions. The approximate mean difference between label 1 and label 0 was 260 transitions.
- More forbidden-key violations, with an approximate mean difference of 5.5 violations.
- A substantially higher number of auto-captured video evidence clips, averaging 21.88 clips per session for label 1 versus 11.83 clips for label 0 (a mean difference of +10.05 clips).
- Higher heatmap coverage, indicating that gaze activity is distributed across a larger portion of the screen.
- Higher entropy, indicating a less concentrated gaze distribution.
- A lower peak ratio, indicating that gaze activity is less concentrated in a single dominant region.
- More time spent outside the center region.

These observations suggest that the label-1 sessions are associated with wider screen exploration, more variable gaze behavior, and more recorded behavioral violations. However, these are associations and should not be interpreted as evidence of causation.

## Feature Relationships

Several features are strongly correlated with one another:

- Entropy and coverage ratio have a correlation of approximately 0.94.
- Entropy and peak ratio have a correlation of approximately -0.92.
- Coverage ratio and peak ratio have a correlation of approximately -0.91.

These relationships are expected because all three features describe aspects of the same gaze-distribution pattern. However, the redundancy may reduce the amount of independent information available to the model and can make feature-importance interpretations less reliable.

## Potential Outliers

Some features contain unusually large values relative to the rest of the dataset:

- `elongation_ratio` reaches approximately 39.65, while most observations are substantially lower.
- `num_transitions` reaches approximately 1,547.
- Several violation-count and duration features also contain relatively high values, including up to 16 forbidden-key violations, 30 off-screen violations, and 24 duration violations in a session.

These observations should be inspected to determine whether they represent genuine behavior, unusually long sessions, tracking errors, or data-processing artifacts. Outliers should not be removed automatically, since unusual behavior may be relevant to the classification task.

## Suitability as a Prediction-Model Foundation

The dataset is a suitable starting point for developing and testing a prototype prediction model. It has a clear target variable, a nearly balanced class distribution, consistent session-level features, and multiple feature groups that describe complementary aspects of gaze behavior.

The current training approach is also a reasonable baseline. It uses a Random Forest classifier, stratified five-fold cross-validation, probability calibration, and a saved feature-column order for inference.

Nevertheless, the dataset is small relative to the number of predictors. With 186 sessions and 79 predictors, the model may overfit, and cross-validation results may vary considerably depending on the sampled sessions. Therefore, the current dataset should be treated as an exploratory or proof-of-concept dataset rather than sufficient evidence for deployment.

## Model Trainer Results

The model trainer was executed using the current complete dataset. It trained a calibrated Random Forest classifier using stratified five-fold cross-validation. The reported cross-validated results were:

| Metric | Result |
|---|---:|
| Cross-validated accuracy | 0.909 |
| Cross-validated ROC-AUC | 0.974 |

The classification report was:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-cheating | 0.90 | 0.91 | 0.91 | 93 |
| Cheating | 0.91 | 0.90 | 0.91 | 93 |
| Macro average | 0.91 | 0.91 | 0.91 | 186 |
| Weighted average | 0.91 | 0.91 | 0.91 | 186 |

The overall cross-validated accuracy was 0.91, with similar precision, recall, and F1-scores for both classes. This indicates that the current feature set separates the two classes reasonably well within the cross-validation procedure. The ROC-AUC of 0.97 indicates strong ranking performance across classification thresholds, although the lower accuracy than the earlier run shows that the estimate is sensitive to the current dataset and should not be treated as deployment evidence.

These results should be interpreted cautiously. They are cross-validated estimates generated from 186 sessions, not results from a completely independent test set. They may therefore be optimistic if sessions from the same participant or recording conditions appear in both training and validation folds, or if any feature is closely related to the labeling procedure. Independent participant-level or future-session testing is still required.

After training, the model artifact was saved as `suspicion_model.joblib`, and the feature order used during training was saved as `feature_columns.json`. These files support consistent feature alignment during session-level prediction.

## Validation Risks and Limitations

The following issues should be addressed before drawing strong conclusions from model performance:

1. If several sessions belong to the same participant, random cross-validation may place sessions from the same participant in both the training and validation folds. This can produce overly optimistic performance estimates. Participant-level splitting should be used when participant identifiers are available.
2. Violation-related features may be closely connected to how the labels were assigned. A separate model trained without violation features should be evaluated to determine whether gaze patterns alone provide predictive value.
3. The labels should represent independently defined ground truth. If the label was assigned because a session was intentionally created as a cheating or non-cheating scenario, the result may measure differences between experimental conditions rather than reliably detecting real-world cheating.
4. The dataset should be tested on new participants and new sessions. Performance on unseen data is more informative than performance on the sessions used to construct the feature dataset.
5. Accuracy should not be used as the only evaluation metric. ROC-AUC, precision, recall, F1-score, PR-AUC, confusion matrices, and calibrated probabilities should also be reported.

## Recommended Future Work

The following steps would strengthen the dataset and the resulting model:

- Collect more sessions from a larger and more diverse participant group.
- Record participant identifiers and use participant-level train/test splits.
- Evaluate models with and without violation features.
- Use repeated cross-validation or bootstrap confidence intervals to quantify uncertainty.
- Test the final model on a separate holdout set collected after model development.
- Investigate the extreme elongation and transition values.
- Standardize recording conditions and document camera position, lighting, calibration quality, session duration, and screen configuration.
- Compare the Random Forest baseline with simpler models such as logistic regression and regularized linear classifiers.
- Examine feature importance using permutation importance or model-agnostic explanations rather than relying only on raw tree importance.

## Summary

The dataset provides a promising foundation for an exploratory gaze-behavior classification model. Label 1 is associated with more dispersed gaze activity, more transitions, higher coverage and entropy, lower peak concentration, more recorded violations, and nearly double the number of auto-captured video evidence clips (21.88 vs 11.83 clips per session). The balanced classes and absence of missing values are positive properties.

However, the limited number of sessions, possible participant dependence, strong feature correlations, outliers, and possible relationship between violation features and labels limit the strength of the conclusions. The increase to 186 sessions improves the sample size, but the dataset is still appropriate primarily for developing a baseline and guiding further data collection. Additional validation is required before the model can be described as reliable for real-world decision-making.
