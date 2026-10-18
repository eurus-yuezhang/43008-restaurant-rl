# Current submission manifest

Prepared on 18 October 2026. This is the current web-only delivery, shared by the
main project and submission folder. Historical names and original records are
preserved in the external archive documented in [the cleanup record](docs/READABILITY_REVIEW.md).

## Contents and validation

- 11 documented core Python modules, one executable Notebook and 27 core test modules.
- 15 selected checkpoint ZIPs: DQN A/C and PPO A/C/G, three training seeds each.
- Web recorder, unchanged recorded traces, page sources and rebuilt HTML.
- Final protocol, final results and architecture-validation evidence.
- 248 retained tests passed; 15/15 model hashes and loads passed; local web replay passed.
- Result values, model bytes and replay data are unchanged. Current model-path fields use the delivery names; comments, docstrings and Notebook Markdown have also been clarified.
- The comment pass preserved executable Python/Notebook ASTs, synchronised 26 embedded definitions and passed 248 tests plus the rebuilt web replay check.
- No desktop demo, caches, temporary validation files, historical backups or previous ZIPs are included.

## File inventory

SHA-256 values below describe the current delivery files. This manifest excludes
its own hash to avoid a circular checksum. Historical manifest fields describe
earlier research snapshots; use this table for current files.

| Relative file | Bytes | SHA-256 |
|---|---:|---|
| `demo_web/build_page.py` | 1264 | `868d9588b0c93f0b0667b398bff94f40db0b4bd2ef15a3a7933bf964573f43d5` |
| `demo_web/README.md` | 1817 | `a33b1a148daad5abd935b3fc8a60e2d62e222a5fddaac723a4a3c582d6ff5501` |
| `demo_web/record_traces.py` | 7940 | `270f9caedfe8d915326ca5b240ffa97c39619bf66098395d2e2c632354740cc6` |
| `demo_web/restaurant_policy_lab.html` | 1453223 | `81787ac277a6c634e44daa9d935ae41991048e35425eebcfd383c287c81e0e51` |
| `demo_web/src/page_body.html` | 3826 | `4252b67fafafcdacb2dbcfec4edf983fcd17ecb81f2aa94bcf2d3afc3e5901b2` |
| `demo_web/src/page_head.html` | 10791 | `3f84c634b7505a09f5ac9c0a87c56acd2326512cba54873a271a93da5debf5f4` |
| `demo_web/src/page_script.js` | 38200 | `27f68edf2577aad132fbdc42e6bcf352155159542b14221aa02ccef1bddac460` |
| `demo_web/traces.json` | 1399947 | `52560c8e63cb7fd239e52182cfd471e8e93e69d9e23234a799785a2d4ff02449` |
| `docs/PROJECT_EVIDENCE_INDEX.md` | 2965 | `91ce122665e1444c65e0c20794b24a8217e345a5fddfaa29f72c8739b98fb5bd` |
| `docs/READABILITY_REVIEW.md` | 7539 | `04bfbf47ed7ab721292085ce25bc3f6512ee0e8e0707ca2a9001d3b3d9e2b69f` |
| `docs/Restaurant_Environment_Specification.md` | 10286 | `38d653fcf96d64083a3df61332fdf3aa0891b86ed10a834b87be3f841e11d2f0` |
| `docs/RUNTIME_AND_DEPENDENCIES.md` | 1255 | `163e282babe724dab7baf1209686fd3ce1f6bf3511c8f9da9a2f7fc7b81e0957` |
| `models/dqn_default_seed31000_best.zip` | 160011 | `abb3312fc04d73c83e910b2a8e5f2dd511a83239afbaeaac3486843526cefd0b` |
| `models/dqn_default_seed31001_best.zip` | 160018 | `84cf1a5de4b8c962897c2cbc09af26413aa397c5b05bd18268fc593d68ea8ebd` |
| `models/dqn_default_seed31002_best.zip` | 160027 | `ad622438f4b11fb5301568da2677de18384f09346ee8f441056d3e10b5dc814e` |
| `models/dqn_mlp_64_seed31000_best.zip` | 91608 | `c1b6f7c705e2b7e69a9f660f7ff31114cd0e0fdca7d78f3c16467bc0975ca5d7` |
| `models/dqn_mlp_64_seed31001_best.zip` | 91612 | `f9f52149a15b2c709da9b15e743e04aab57cb7438929641161a834ac1e80ff9f` |
| `models/dqn_mlp_64_seed31002_best.zip` | 91595 | `9a10adc6c8a8cdacf2ec53699dff337222d4edc42aa4768fc1236f1e1af356c9` |
| `models/ppo_cross_attention_seed31000_best.zip` | 90306 | `ab3650519430ea98f69fd8a65a2eb8bcbffe39ea3f28ed4025e51622a88c72f3` |
| `models/ppo_cross_attention_seed31001_best.zip` | 90310 | `be72014fcbe437bbd36d1c4981b24a0fd74ca3cbbf3f6b4a771cd3907c879c57` |
| `models/ppo_cross_attention_seed31002_best.zip` | 90290 | `a2b76f0be20c7ab0ff1fd74ec85b5708cdf342975037bacb433b171fdd7447cf` |
| `models/ppo_default_seed31000_best.zip` | 221225 | `66a782f50654909849e72a48face7064a8f9acabf58f555a55431d1aa9680ea2` |
| `models/ppo_default_seed31001_best.zip` | 221245 | `e307376fdb4348ac2316763bf1b305e41c63c4b78e79529867b3a77447aafb54` |
| `models/ppo_default_seed31002_best.zip` | 221225 | `1d2b50d1cda12831046a9c34872549c590f570e5ee3dba226a64c0d9ce4bc90e` |
| `models/ppo_mlp_64_seed31000_best.zip` | 76578 | `9f6f856f648926a3905547b3e03b485ff2e8f30edfc8071d476545c14367cd1e` |
| `models/ppo_mlp_64_seed31001_best.zip` | 76582 | `e2f1e9c8f0988d22d4b917dd495f68c8f26c3b785b9238261f61b9ddf96c36be` |
| `models/ppo_mlp_64_seed31002_best.zip` | 76578 | `6893ea4bcd0a64f7aaf0c33778bdc8754a38464d73e46d9b78702596ab16f300` |
| `notebooks/43008_Restaurant_RL_Final_Executable.ipynb` | 516807 | `0cb87c44b1bfb998f852d2890a21fb64c57a54715e3f3b06ff25b53fe644d4f3` |
| `README.md` | 3956 | `0eb1a77ed05eb9841db2b736534015186bc6645ea25b3dd1e4e6d0dbf6830d8a` |
| `requirements.txt` | 301 | `8e8fa24c48af00e36523521fec6b8ea1a02f3515f0614cfb619367b16c731765` |
| `restaurant_rl/__init__.py` | 83 | `d94bdc6894ed4665dddd9880cd767fdc8c3f3af9ca88b499f274fe9bf82de049` |
| `restaurant_rl/action_masking.py` | 3042 | `3038ff7d97c2ea8fc95c86a0b8e7f562f327e2a22f08e20a093415fc26b83464` |
| `restaurant_rl/architectures.py` | 2227 | `d95083c0906b85c033ba452420c4b46147484b7c301274222a217d2b871353c9` |
| `restaurant_rl/baselines.py` | 2107 | `08bc99e3c233eb7ac40417a17748b9ab8362bc28e6b1885a48df91fc9c0c3979` |
| `restaurant_rl/config.py` | 6987 | `377247a5633f1400f20e0caedd0117c0a7aad170a9f7e2df05802d915d1873f9` |
| `restaurant_rl/entities.py` | 2955 | `7f9541e9bbe446ef53326d6d5e39c68a1c6b722e691b146a3650d5b9b59f944b` |
| `restaurant_rl/environment.py` | 19124 | `0015092555c4a4bd1198344d2f6c8b59f51c832504afdb8456abd3c91ed07637` |
| `restaurant_rl/feature_extractors.py` | 16680 | `813ea7d0a48d1a6980ca8cfc0ab8746ec1ecb1c36c16e98ac0cdb8b0d081c457` |
| `restaurant_rl/masked_dqn.py` | 3287 | `a0e2e40aeae8dae8815b299c8ecfb6732a3ae31ea261573c81f142fc63622ba2` |
| `restaurant_rl/mixed_training.py` | 3495 | `1865c3251d1223c4bf53186e3ddea4f5eeda721439f6c0a9c7527861304436d2` |
| `restaurant_rl/randomness.py` | 1099 | `c3b540874c2c11195ad2bbf153ae3808bd0f4c3b982f8ed77eb00733ecfe463c` |
| `results/architecture_validation/A-E/architecture_comparison.csv` | 4920 | `419926713bc4c0267658dc92072fe6ce3e811310abe62ae8580f3ffafcb50efe` |
| `results/architecture_validation/A-E/per_seed_summary.csv` | 6362 | `e00adf0838994cd14005de2876fb9727deee5b198bba9ea4f45623e7d7d83d46` |
| `results/architecture_validation/A-E/study_manifest.json` | 1727 | `ee56ade1be38270575fd75fa857e32f20550de48835aeffc8ee8cff488e49b95` |
| `results/architecture_validation/F/controlled_comparison.csv` | 5510 | `2f9777f82e9b746ef2cb50f825248db495d976fffd736d4ce3f7fc226ee3b04d` |
| `results/architecture_validation/F/structured_architecture_summary.csv` | 3631 | `e533782462fac2430203aa1f33e29bb69f96667584b8f44792b99854af86ffc7` |
| `results/architecture_validation/F/structured_per_seed.csv` | 4659 | `e3b28c27b796dcbd8fcabd5fad3c44ffa45afab172b202e6f57ddadb9771dd6f` |
| `results/architecture_validation/F/study_manifest.json` | 1927 | `1340c00252388b56c7cf36f0648c58d73749ebac827bc0c988658116e1dd5b43` |
| `results/architecture_validation/F2/controlled_comparison.csv` | 3647 | `c8653645fddb9f7a85782d1c380d61febd5fbfdc18e8b8e9a87e3b82017f0b79` |
| `results/architecture_validation/F2/f2_per_seed.csv` | 2052 | `0b4ac289b691bb6b6bd11e0b6e866c13feacef9c8d9e56808ed802c5d8869e70` |
| `results/architecture_validation/F2/f2_summary.csv` | 1693 | `a4701ab32fa0647d06a5636755076d83eed92d6b19480ba9898a875efdf7386b` |
| `results/architecture_validation/F2/study_manifest.json` | 1988 | `3d75ab8694b3e883072ed01a9ce02bd4264358182865110d1da7c970e2c056a6` |
| `results/architecture_validation/G/attention_diagnostics.json` | 6362 | `590af1cad69d59815e02af4b3cb7b846f809a83ac99fdb14a6b4bf3e64dd04ce` |
| `results/architecture_validation/G/controlled_comparison.csv` | 4288 | `4b1da0a87f80d4c2735741e9b0f19556de3c38f108a57d37e184a5af1a1bbcdf` |
| `results/architecture_validation/G/g_per_seed.csv` | 2166 | `72b1d2ccd1da829c2d40e0f34a65dcd2b06bf05c91251483b5c62a49fec518d6` |
| `results/architecture_validation/G/g_summary.csv` | 1796 | `0326f3c8795ada30eb7019ef326ce57593dab434ae2c25ff58f6d1b1d9ce25cd` |
| `results/architecture_validation/G/study_manifest.json` | 2182 | `28d93aecc4821a26f0f9f7113ead9b1475017c8f4969aa850d166c5667ff79cb` |
| `results/final_evaluation_freeze/candidate_unused_seed_ranges.csv` | 986 | `bd269073130be0d9308290f250277f7df0b5f79e60b75d3ea36ee8f7038e6e6d` |
| `results/final_evaluation_freeze/final_comparison_plan.md` | 2470 | `54dfea056748a84df988a417a8ab8dcce17fd109d56bbac0e8922fc06b5e738f` |
| `results/final_evaluation_freeze/final_metric_spec.md` | 2906 | `5fa19340b8e07518c5e37c133e1a4615d226200732d52fe23bfad353e6055386` |
| `results/final_evaluation_freeze/final_model_roster.csv` | 6078 | `10fc9bb424a8d38402668555523837347784f350ae2b9096c8459db69be47efc` |
| `results/final_evaluation_freeze/final_seed_freeze.json` | 2424 | `1e84f77599b7e8a6f24444d5e75512c829ca8c97055ce7405a0524279d8b2b05` |
| `results/final_evaluation_freeze/freeze_manifest.json` | 67125 | `0af35b4d466c786f3317a7f6b3906bbc64f6fa5133483bac01f97b55d30277f2` |
| `results/final_evaluation_freeze/seed_ledger.csv` | 7483 | `4f88184afd88e5aee1ca006dc42ad02ae5a5cfa24f84e06f0654640dda825c67` |
| `results/final_heldout_evaluation/final_architecture_summary.csv` | 3242 | `5b6e7d66aac96dc603dfe5d2939169d8d9a08590c049e70996238dbc8fc5b35b` |
| `results/final_heldout_evaluation/final_episode_rows.csv` | 2849143 | `bb019970de011069ac1e2924a180056d4b6faaa1ba0db89906ff6a4eb8e24675` |
| `results/final_heldout_evaluation/final_evaluation_manifest.json` | 30599 | `e78fa8fcd5e89a1414b22766f8887980c652de4db2250f962658eedbeadbaf35` |
| `results/final_heldout_evaluation/final_model_seed_summary.csv` | 6148 | `1576b73c9f6c40dabc2a423398b2725dd88f26e1f78b101630f95158d90b4ef7` |
| `results/final_heldout_evaluation/final_paired_vs_bestfit_architecture.csv` | 849 | `9f346ef4c0114a7d8a9f2c5f7dc24de7a2bde9c5ddeadd88bbf791df3b79dedc` |
| `results/final_heldout_evaluation/final_paired_vs_bestfit_per_model.csv` | 1479 | `e6eb6a074bc8415afb7d3b433f4813177a3afa3da0ebdcd0953df0dc9e63a051` |
| `results/final_heldout_evaluation/final_preregistered_comparisons.csv` | 7583 | `347d3ff679c3d1ecf94f5f28d514fbf312609a3326fd82baf2b3701acbbda038` |
| `results/final_heldout_evaluation/final_results_report.md` | 5409 | `8a2944bf731b60de5ac5d7f250b6eff45e0003578de993bfc81aa185069c2a41` |
| `results/final_heldout_evaluation/final_scenario_summary.csv` | 10761 | `194920522c940964ecc2cf4742abdc2d54d1ff4096c23cf4ca2df7fec138744b` |
| `tests/test_abandonment.py` | 2423 | `80373586c0667b174e82be492f56b5bedb55ba51ea90950fc6da543d9ebbac59` |
| `tests/test_action_mask.py` | 2882 | `d05251fe30170498c39dabf52311a639859017cc4ea86ed5262ce3a94367032b` |
| `tests/test_arrival_boundaries.py` | 1794 | `f87e6451cfa26a37bb421603c6aab9891561ab993804811de298dd8fc15ddf50` |
| `tests/test_arrival_reproducibility.py` | 5652 | `6ca8490483de03fc434790e90727f8ef7f34834ee285603ad5f59ba0ad25bc14` |
| `tests/test_arrivals.py` | 3613 | `51b7979b83db78eb2d266f34690d78d228b3c0b15a092efe6ffd9a80671d384d` |
| `tests/test_baselines.py` | 2209 | `6ae542442c5503ba6be58a92b7930e20ee2c8ac9d9b99ff77e16847a090da962` |
| `tests/test_buffer_invariants.py` | 6092 | `9bb6c3afd9a47de0c49603da09dd002c4fe4314d0a9ad2937ef90850812ffeec` |
| `tests/test_buffer_lifecycle.py` | 3638 | `05eb0422a6e58804b1a7b82b6a3ddbd9305127b9c6b1ba048dc888162d56745c` |
| `tests/test_closing_lifecycle.py` | 2375 | `598e1154667ae777d0e21df6ac192a7727b76ec79d4a69fbf7216af3cf23ceed` |
| `tests/test_config.py` | 5955 | `f470923299227e830f00399e8a40a1caae84fdb835606834dfd860c5598531ba` |
| `tests/test_cross_attention_feature_extractor.py` | 7347 | `108e6cdf8e6c2f78aa0d9b37c24949ded629ac3714f28b1f5870c21355edc894` |
| `tests/test_dual_structured_feature_extractor.py` | 8608 | `5fdca348b9000298aed045554e3e441139cac1c2fbf65ed81d43bb71c425567a` |
| `tests/test_end_tick.py` | 2900 | `e59129b0dd02083a4aabebbc75644a583b59042f8d49e8ad81ff62e7d08de171` |
| `tests/test_entities.py` | 2372 | `c7f121768e08c7e5a304d19734bb40b5a06f3a2f94f80d10b199808467aaa885` |
| `tests/test_environment_reset.py` | 3582 | `2534d790b83fea0b39da72280544413843081b29ae034eef60e993102a659951` |
| `tests/test_feature_extractors.py` | 4971 | `5228f19e09545fa7b8dfb14bc8dec283d7291be7b5571d9b4ad34f109579ecc4` |
| `tests/test_keyed_dining.py` | 1725 | `faa90041f6936af94656ecbf051e57908793713be521fba40519aecc9c519aa1` |
| `tests/test_lifecycle_smoke.py` | 2037 | `0d5f167c066b3fc34d064b7a8ee73979c4d345b9f381bac4857e1b12dfbf5a35` |
| `tests/test_mixed_training.py` | 2041 | `dfd38e29a92666b50997060851dcfd7cadccbd4a3972d8bd086dfb40d74077ed` |
| `tests/test_observation.py` | 5044 | `3821483817aa5609ce706dd48de5989e2b07f8f2dae71070b6d3c8a1a525fe15` |
| `tests/test_patience_schedule.py` | 6284 | `873bbd283ffc4f81e3ff69813fc46603e4affe46116392006d94131299ae3218` |
| `tests/test_reward_accounting.py` | 6623 | `0458d7888ec1034a63e5eabbccd46030c5533ef4d8be6305ec55e39042c76e5c` |
| `tests/test_reward_reproducibility.py` | 2191 | `b01aa6a22aed1399fa69fbb84261fe49cb909ff5c275516b22b39c422a100f6a` |
| `tests/test_rl_masking.py` | 2843 | `d5e657db40f9aad5316be728d9c9d311a57bf84a7c9f74f7ec2ed6c817b136fb` |
| `tests/test_seating_actions.py` | 4843 | `d265ea9e7a37b67df7fff36dee7cfdc1923e6c1b9817a521cfa9ff27b3c573c1` |
| `tests/test_structured_feature_extractor.py` | 6763 | `7d525425b14c3ee12f0215cf6487d8cd9ce773e42112e3576f2ffb2830c6adc9` |
| `tests/test_termination.py` | 7365 | `730bd678037bf44f5d61fd922e7c9c302a1dc1be4691b63e97e94ce35d56ce41` |
