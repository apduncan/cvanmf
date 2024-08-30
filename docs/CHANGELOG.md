# CHANGELOG

## v0.2.0 (2024-08-30)

### Documentation

* docs(*): micromamba troubleshooting, docstring

Fix method to make conda available to nextflow when using micromamba.
Update some docstring formating to prevent newlines.
Add semantic versioning to dev dependencies. ([`6bc8093`](https://github.com/apduncan/cvanmf/commit/6bc80936ae586386336ccdafb8ba94c144129bcf))

### Feature

* feat(denovo): plot_metadata now can include modelfit

add model fit to the standard metadata plots. this will eventually replace plot_modelfit, but for now both exists. ([`36e1f7d`](https://github.com/apduncan/cvanmf/commit/36e1f7dc945f25193ee96bfb019686886fe4f092))

* feat(data): nsclc example data and notebook added

add case study on NSCLC atlas to documentation, and included data in the data module. ([`010a8e5`](https://github.com/apduncan/cvanmf/commit/010a8e59b15687c11b78c33fe3668cd5391305a3))

* feat(denovo): stability rank suggestion

add function to suggest ranks based on peaks for stability based measures (dispersion, cophenetic correlation) ([`5447016`](https://github.com/apduncan/cvanmf/commit/5447016086f079929ada01eeae3013bcc6f84d13))

* feat(denovo): change plotting backend

change relative weight plot to use marsilea
remove patchwork lib
update plotnine to 0.13.6 ([`b095e84`](https://github.com/apduncan/cvanmf/commit/b095e840dcd082c78311ca71ae568b251bf563bd))

* feat(data): data subpackage added

Add data subpackage, along with ALL/AML dataset and Swimmer dataset.
Move synthetic data generation method to data subpackage.
Add ExamleData class to hold data and metadata for provided/synthetic datasets. ([`ec89c6b`](https://github.com/apduncan/cvanmf/commit/ec89c6b680b7ef5f3f29abbd9c8d7e0d126e3939))

* feat(denovo): Stability rank selection methods

Add cophenetic correlation and dispersion coefficients for rank selection and associated plots. ([`c780607`](https://github.com/apduncan/cvanmf/commit/c7806077d14840264b8eb074b40ab087e7ccc95e))

### Fix

* fix(denovo): remove rank 1 from stability rank suggestion. ([`1829a17`](https://github.com/apduncan/cvanmf/commit/1829a174e001b4e01dd86f98920a3e3bf267dfa0))

* fix(denovo): load from tar.gz properly ([`8fd938e`](https://github.com/apduncan/cvanmf/commit/8fd938e03dfce95bc0f5f4241d2fac604f47dc2f))

* fix(denovo): allow custom colours for group in relative weight plot

`plot_relative_weight` now accepts parameter `group_colors` which is a series mapping metadata value to a colour for the ribbon element of the plot. can be blank for default Marsilea colours. ([`3cf28ee`](https://github.com/apduncan/cvanmf/commit/3cf28eed3d08c7880bdcf327105a24fc5345d959))

* fix(denovo): use consistent colours for bicv measures

`plot_rank_selection` and `plot_regu_selection` now uses the same colour for each property whenever displayed, doesn&#39;t always start at the same colour. ([`bf8f91b`](https://github.com/apduncan/cvanmf/commit/bf8f91ba26acc1b00dfd2a8525120d348ff6cb2b))

* fix(denovo): change to star suggested rank/alpha

`plot_rank_selection` and `plot_regu_selection` use a star at the top of the plot to indicate suggested rank rather than a dashed vertical line. ([`6a23b2d`](https://github.com/apduncan/cvanmf/commit/6a23b2d9e203ffa21dbb7308d260e74a1b00897e))

* fix(denovo): improve handling rank 1 by stability

removes decompositions of rank 1 from calculation of stability based rank selection methods, as not meaningful. ([`ac80647`](https://github.com/apduncan/cvanmf/commit/ac806478158fa8628a2260d0897923d7a92dace4))

* fix(data): include swimmer data

added swimmer data to repo. ([`9ef625b`](https://github.com/apduncan/cvanmf/commit/9ef625bd1a6131ea29599d6d6a16da27acf7fd5f))

* fix(*): tidy loggers

use module loggers rather than the global logging functions ([`d918057`](https://github.com/apduncan/cvanmf/commit/d9180573d8bc4ce80e06d150a18db17fbd8b343f))

### Unknown

* Fold design (#9)

* feat(deono): design parameter for bicv

support different designs for bicv. `rank_selection` and `regu_selection` (and all underlying classes/functions) now take a tuple parameter `design`, which determines how many times to split the shuffled input during bicrossvalidation. you are no longer limitted to 3x3 9-fold cva! ([`04ab493`](https://github.com/apduncan/cvanmf/commit/04ab493d2ad5dfcd9e74246fada049451d41c893))

* Merge pull request #8 from apduncan/modelfit_plotting

Modelfit plotting ([`3fa104a`](https://github.com/apduncan/cvanmf/commit/3fa104ab4671d802cac60581e3b2d8637c857690))

* metadata accept df. need to extend to univar tests. ([`1cd83c1`](https://github.com/apduncan/cvanmf/commit/1cd83c1a23071ea45553e90b623d7babe924e339))

* Merge pull request #7 from apduncan/cell_example

feat(data): nsclc example data and notebook added ([`e070d61`](https://github.com/apduncan/cvanmf/commit/e070d616493e2c4f46bb8a726a715fb4d3646565))

* Merge pull request #6 from apduncan/suggest_peaks

feat(denovo): stability rank suggestion ([`62552fa`](https://github.com/apduncan/cvanmf/commit/62552fa328117766a497ff2be234253bda7ddc45))

* Merge pull request #5 from apduncan/replace_patchworklib

feat(denovo): change plotting backend ([`cd4c2ed`](https://github.com/apduncan/cvanmf/commit/cd4c2ed34ba9ba7c8523a93aa2891c0823076be9))

* remove patchworklib dependencies and import ([`461e17f`](https://github.com/apduncan/cvanmf/commit/461e17f05c792a1b2fcb6ef3a966befb274da1f6))

* merge logging changes to marsilea work ([`9f52d34`](https://github.com/apduncan/cvanmf/commit/9f52d345a32e4181f59f5777e6b594cde80a58d9))

* Initial work on replacing patchworklib ([`fb58280`](https://github.com/apduncan/cvanmf/commit/fb58280dd45546779ecccab9c9dadd8e2c9674ca))

* Merge pull request #4 from apduncan/logging

fix(*): tidy loggers ([`372d4b3`](https://github.com/apduncan/cvanmf/commit/372d4b303cde4a96fc937429f82b847efc2dad56))

* Tests for data ([`24aa1bc`](https://github.com/apduncan/cvanmf/commit/24aa1bc918855eaf8109214c41f7ae2569e8ff19))

* Initial work on replacing patchworklib ([`8480321`](https://github.com/apduncan/cvanmf/commit/8480321015654353e79fb180af1341ac2b9c344c))

## v0.1.0 (2024-07-18)

### Unknown

* Generators for BicvSplit rather than lists, reduce memory for large inputs. ([`4cd269c`](https://github.com/apduncan/cvanmf/commit/4cd269cf55fec1f8830b0eecf8861340ae736ae5))

* Change default plot behaviour for decomposition save. ([`0669dd7`](https://github.com/apduncan/cvanmf/commit/0669dd7c677499e1f74931df23481f2f2d62bc2b))

* Amend nextflow docs ([`b749d8e`](https://github.com/apduncan/cvanmf/commit/b749d8e43ed838c161306159943bb9d01952140f))

* Relative dir for slurm env.yaml ([`0af5f82`](https://github.com/apduncan/cvanmf/commit/0af5f82ea4b5d4a7b2043afd6a544f640cad9a83))

* Nextflow documentation ([`5e47929`](https://github.com/apduncan/cvanmf/commit/5e479295b8d1eacaa42f3d5ad0a38d5d691b8072))

* Documentation formatting updates ([`8fcab72`](https://github.com/apduncan/cvanmf/commit/8fcab721766504cb8a9e9de9bdf4d0251ee18306))

* Minor fix: nan suggested rank in plot ([`6d8634f`](https://github.com/apduncan/cvanmf/commit/6d8634fff9c8c826e68b5cd234cfa3f554e1f95d))

* Fix for docs: add furo to reqs. ([`ac4a5e2`](https://github.com/apduncan/cvanmf/commit/ac4a5e252dca00a06dedb1ffa9bd5bcac5362323))

* Docstyle update ([`b37cfd5`](https://github.com/apduncan/cvanmf/commit/b37cfd56cb17f345d0d036f52df9e5aa005f116e))

* Command line docs update ([`40c5b02`](https://github.com/apduncan/cvanmf/commit/40c5b026d84a772f0203b217ace0659d58c8770d))

* Install instructions ([`c49ef86`](https://github.com/apduncan/cvanmf/commit/c49ef860e49ddaf718524ec1f7bab14748bcb334))

* Extract alpha suggestion to function ([`7e8a76c`](https://github.com/apduncan/cvanmf/commit/7e8a76c3a8ed38307c7d22d65f37807d7b7fe334))

* Fix rank selection plot when no elbow detected ([`c156f7d`](https://github.com/apduncan/cvanmf/commit/c156f7deb40d67cb400e26b661c353a4e297e1c3))

* Significance annotation on metadata plots ([`d610cc8`](https://github.com/apduncan/cvanmf/commit/d610cc8b74fb5372f73c0537ce2c7881f619e9a4))

* Automated elbow detection. ([`aaf1664`](https://github.com/apduncan/cvanmf/commit/aaf166469ab2e3ec5cddc9dfc99238f723c7399e))

* Regularisation selection CLI ([`b9d0f4f`](https://github.com/apduncan/cvanmf/commit/b9d0f4f7c009e1a01bac2e7e4f427d445b3a14b1))

* Misc changes ([`2660120`](https://github.com/apduncan/cvanmf/commit/26601205b7e285865277400cab0ae9bd63abc3ee))

* Doc build fixes? ([`774c80f`](https://github.com/apduncan/cvanmf/commit/774c80f43ab527af8b5e0fb1c81ef1a39d6a708f))

* Doc build fixes? ([`3191b94`](https://github.com/apduncan/cvanmf/commit/3191b9459fb6c3a5a20875232098b89b99c99f86))

* Doc build fixes? ([`7a9140c`](https://github.com/apduncan/cvanmf/commit/7a9140cc93c159b22d95efbae6feed4e8d87eaff))

* Doc build fixes? ([`be335d1`](https://github.com/apduncan/cvanmf/commit/be335d186953ab16ee90f37183c3a3c89702ef48))

* Doc build fixes? ([`9181f34`](https://github.com/apduncan/cvanmf/commit/9181f34d9af453101c96930afa6fa7d0d0e1001a))

* Add sphinx click ([`d964b65`](https://github.com/apduncan/cvanmf/commit/d964b6582e918a055d7f923d101d27c45f25121e))

* Doc build attempt ([`52c3e3a`](https://github.com/apduncan/cvanmf/commit/52c3e3a349709728cad2884fce373f6016d2608b))

* Merge pull request #3 from apduncan/reapply_refactor

Reapply updates, combine methods added, other changes ([`b497ee0`](https://github.com/apduncan/cvanmf/commit/b497ee0ec1fdca3a98a16ee8eecb5583c4c70395))

* Plotting improvement, scaling synth data. ([`0d3df67`](https://github.com/apduncan/cvanmf/commit/0d3df67807eb3c8d3cb62cbea29434008bc8d36c))

* Univar tests, plotting, qol. ([`abe62ab`](https://github.com/apduncan/cvanmf/commit/abe62ab12e7d0626eece0a222be1df92653d274d))

* Plotting improvements. ([`5fae847`](https://github.com/apduncan/cvanmf/commit/5fae847c64c19d0c365569fe7e731a5738d12360))

* Add PCoA. Remove MDS. Tests. ([`7fd2999`](https://github.com/apduncan/cvanmf/commit/7fd2999a1261a79b13a08534a46e029780a5fd27))

* Regularisation selection ([`94141b7`](https://github.com/apduncan/cvanmf/commit/94141b7e6089dc634a6729532825905865580ef7))

* Additional work on combining ([`9a8a722`](https://github.com/apduncan/cvanmf/commit/9a8a72236f84a405dfcab7882587fdbc1549eda2))

* Basic approach to combine models implemented. ([`fc582a7`](https://github.com/apduncan/cvanmf/commit/fc582a79e5a90458784b90dd72aa52f73d6814b0))

* Bugfix: Do not summarise i ([`3220efd`](https://github.com/apduncan/cvanmf/commit/3220efd7776907b3e15dbe29572c6a16fd1eca41))

* Prevent copies of X matrix in BicvResult ([`bca551a`](https://github.com/apduncan/cvanmf/commit/bca551a3313945af469470863dad48a36a7c70d3))

* Passing test ([`e4b6f2b`](https://github.com/apduncan/cvanmf/commit/e4b6f2bfb63e3df4b6304fb6a9038e91c89d50a2))

* Passing test ([`29d7fa8`](https://github.com/apduncan/cvanmf/commit/29d7fa8c4d33595ac81b876e4cff25445f9b3614))

* Generalised. Import problems. ([`b694103`](https://github.com/apduncan/cvanmf/commit/b694103aa9f45f8ead5d9118e4c1dcfe843979f0))

* Start generalising reapply ([`ab0f542`](https://github.com/apduncan/cvanmf/commit/ab0f5425e117ce4f4a356d2256b6ab9ee5189507))

* Fix distutil? Shorten tests ([`3115785`](https://github.com/apduncan/cvanmf/commit/3115785d9621882effbf90bc798cee418024d458))

* Allow manual ([`b318b1e`](https://github.com/apduncan/cvanmf/commit/b318b1ec07b6dc4009455217ed984b28090a5812))

* Merge remote-tracking branch &#39;origin/main&#39; ([`4850b22`](https://github.com/apduncan/cvanmf/commit/4850b22db5a7d4324b7726e166c5c5a3324ad721))

* Merge pull request #2 from apduncan/package

Start CI impl ([`f8913c3`](https://github.com/apduncan/cvanmf/commit/f8913c3db0dc1953ac094e33843a6fe597b8baa8))

* Merge branch &#39;package&#39; ([`304b1ed`](https://github.com/apduncan/cvanmf/commit/304b1ed6e1544e38cf617416b68ca536ecbe452b))

* Allow manual ([`e188417`](https://github.com/apduncan/cvanmf/commit/e188417251c355e06010386bf5fe74e6077e7958))

* Start CI impl ([`cce5173`](https://github.com/apduncan/cvanmf/commit/cce5173c89426fc4c99f0c7d8dd29f9209e05382))

* Merge package structure to main ([`ca46200`](https://github.com/apduncan/cvanmf/commit/ca46200c331edcb1f93ddb332fb94c9a6c87e790))

* Separate SL and package; rename ([`167b142`](https://github.com/apduncan/cvanmf/commit/167b14250e0f9d88ed382c212baaf04f509ae692))

* Slicing implemented. Plot fixes. ([`fada983`](https://github.com/apduncan/cvanmf/commit/fada983c760ccbd9906b4878c428e717417ac930))

* Plotting and doc updates ([`b171f20`](https://github.com/apduncan/cvanmf/commit/b171f20fe929587b723d7e32541237233a5361ab))

* Plotting and doc updates ([`e057a62`](https://github.com/apduncan/cvanmf/commit/e057a6269c87ee2d2236079632152e143ebce05a))

* I/O for decompositions &amp; tests ([`89896d1`](https://github.com/apduncan/cvanmf/commit/89896d18e7b027c6e419de753362a0d17078a002))

* Minor log format update ([`258eb38`](https://github.com/apduncan/cvanmf/commit/258eb38ce9e4e2c74ac8dc944582dd8489ffe2b0))

* CLI tests ([`3d16cda`](https://github.com/apduncan/cvanmf/commit/3d16cdaaf8364101bad349d1539c381c83d795ee))

* Test suite improvements ([`1ff2546`](https://github.com/apduncan/cvanmf/commit/1ff25466af136e1624dbb21c250e73fd5355659f))

* Some data generation ([`8f7bb99`](https://github.com/apduncan/cvanmf/commit/8f7bb99797faef9b63285f70f00744f8e14735d8))

* Additional plotting ([`917baa1`](https://github.com/apduncan/cvanmf/commit/917baa12a7c1904558de973edfbabe515bcb460b))

* Model fit plot ([`08379cc`](https://github.com/apduncan/cvanmf/commit/08379ccb88ef76ee16fae7680d21463a46d37443))

* Decomposition wrapper ([`e97b766`](https://github.com/apduncan/cvanmf/commit/e97b7661af24bb841b5b58b8adf02f8671e5a2d9))

* Minor ([`f3d9912`](https://github.com/apduncan/cvanmf/commit/f3d99127aaba945fa0350411a40785b796e22cf8))

* Rank selection local. CLI docs. ([`3c50f4f`](https://github.com/apduncan/cvanmf/commit/3c50f4f76df3ef00210d8a88626c71f5ba9e6efc))

* Minor gitpod setup ([`89436d8`](https://github.com/apduncan/cvanmf/commit/89436d823f90a81dc6404c7c7926737b07bdc499))

* Bicv implemented. Rank selection started. ([`9c24801`](https://github.com/apduncan/cvanmf/commit/9c24801a4dc31b1987803564bd152e337bdfd2ad))

* Rename transform -&gt; reapply. Start denovo. ([`b845e8f`](https://github.com/apduncan/cvanmf/commit/b845e8f340d225f9eb951c574bd76cb5e3f59ab2))

* Improve documentation ([`44584f4`](https://github.com/apduncan/cvanmf/commit/44584f42f9b817787553db5c17ec7ae5cd5c3311))

* Structure, unit tests. ([`a9c85be`](https://github.com/apduncan/cvanmf/commit/a9c85be3b8278aabb4ad89f97492dcf5b1798690))

* Started proper package structure ([`9f22f74`](https://github.com/apduncan/cvanmf/commit/9f22f748e87e8605f5e6a5fe3ab52a098386bbf8))

* TSS H/W matrices. TSS H for plot. ([`b998511`](https://github.com/apduncan/cvanmf/commit/b998511e6ac0ec18e2858b0fe0e85e9b34c79310))

* Single method transform for calling from python/R. ([`ddcf864`](https://github.com/apduncan/cvanmf/commit/ddcf8642552e6fda07bfd3afc764560797217890))

* Actually use the family toggle ([`51ba8d0`](https://github.com/apduncan/cvanmf/commit/51ba8d0ef6a0910e9f3669b41bb814d6698f16b8))

* Family option; lineage cleaning ([`84f8431`](https://github.com/apduncan/cvanmf/commit/84f84317b7919ce0b0b26ee91aec7c18206cbef5))

* xsrf fixed, warning removed :) ([`6d6ae30`](https://github.com/apduncan/cvanmf/commit/6d6ae30c34964721bc7e017d0209b7d97f460525))

* xsrf fix ([`e614d00`](https://github.com/apduncan/cvanmf/commit/e614d00e89e6294b75fe9cf9d7a2009423b65b4e))

* Hosted url ([`3684c19`](https://github.com/apduncan/cvanmf/commit/3684c19ce2d964d46dfd8c7d014b189788fc1d7c))

* README ([`1adb8bb`](https://github.com/apduncan/cvanmf/commit/1adb8bba1c4806387cf42e30aaba0137bd8af8b6))

* Command line, NamedTuple ([`a7d10ca`](https://github.com/apduncan/cvanmf/commit/a7d10cac27571fcb5dcaf1ab03e536e849503809))

* typo ([`c1769d4`](https://github.com/apduncan/cvanmf/commit/c1769d4472a45bc9b884e558d50c0587f8c3d28d))

* Log in zip, organised page ([`aca39a2`](https://github.com/apduncan/cvanmf/commit/aca39a2ee1ab7b10c9d338170007ed374c8c6b82))

* Brief result description on main page ([`990c3e0`](https://github.com/apduncan/cvanmf/commit/990c3e08ea047c9ae20ba84170e6525bb3d14cf8))

* Basic caching added ([`0a3e185`](https://github.com/apduncan/cvanmf/commit/0a3e1853f1ad626fb85e6e2a67c3fb606ea2bba9))

* Upload warning added ([`8d50a0f`](https://github.com/apduncan/cvanmf/commit/8d50a0fe9d6e069d081ca93f0e6ad6d84ec18f93))

* Zip results, histogram, model fit ([`34b44c8`](https://github.com/apduncan/cvanmf/commit/34b44c81eb8372e156c0e7eb064d1f44c5ca54aa))

* Revise ([`d89c421`](https://github.com/apduncan/cvanmf/commit/d89c42162d29b722165a3803dc290a1fec50378f))

* Offline text change ([`722352e`](https://github.com/apduncan/cvanmf/commit/722352ed10a01081d6e71c4c9492e72580e97185))

* Fix for MF -1 row ([`4b68495`](https://github.com/apduncan/cvanmf/commit/4b6849555b89431e167da828a19ddca7c39f3fe8))

* NMF param fix ([`9a010df`](https://github.com/apduncan/cvanmf/commit/9a010df3447494f597e84e8dd91a78086a33fb41))

* Change data location ([`fa7bcb4`](https://github.com/apduncan/cvanmf/commit/fa7bcb40702754a57df3e5b78c004bc0a9f096f7))

* Requirements again ([`7ed8447`](https://github.com/apduncan/cvanmf/commit/7ed8447df028f6bbbc45a1c23b79d69c4e7c027c))

* Requirements fix ([`30d9941`](https://github.com/apduncan/cvanmf/commit/30d99417251252cc7cb5ea24ba15d933ac811799))

* Initial commit ([`233572e`](https://github.com/apduncan/cvanmf/commit/233572e7bafe46a6a97af9c82a075528e377fff4))
