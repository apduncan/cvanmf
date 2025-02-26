# CHANGELOG


## v0.5.0 (2025-02-26)

### Chores

- Add ps to nf container
  ([`a91da23`](https://github.com/apduncan/cvanmf/commit/a91da23d37ff09eae202d189a0d28526fd7a30c9))

Added 'ps' to Nextflow container, which is required for Nextflow to generate runtime and resource
  usage reports.

- Fix build trigger
  ([`a6d80a8`](https://github.com/apduncan/cvanmf/commit/a6d80a891bd752e2d0096fbadecd3754ab8c705f))

Typo in actions for triggering PyPI build/deploy

### Documentation

- Clarify reapply input & processing ([#15](https://github.com/apduncan/cvanmf/pull/15),
  [`8921280`](https://github.com/apduncan/cvanmf/commit/8921280e444e6aea4012455d977bee9743b07af4))

Making it clearer that five_es reapply expects DataFrame, and will total sum scale data. Also bumped
  python to 3.12 for itertools.batched.

- **cell_example**: Update figures to fit manuscript draft
  ([`3427a96`](https://github.com/apduncan/cvanmf/commit/3427a968eb1d46f582ea2a7b089eb38064fd4072))

- **README**: Add references to readme
  ([`1fdeb6d`](https://github.com/apduncan/cvanmf/commit/1fdeb6dd71bf4816a4a55bdc2d7497af25316740))

### Features

- Add pypi deployment to CD ([#17](https://github.com/apduncan/cvanmf/pull/17),
  [`9a455a3`](https://github.com/apduncan/cvanmf/commit/9a455a369457a555a3039a8860b2a524a0111637))

Added building and releasing to PyPI to github actions.

- **denovo**: Added plot of weight distribution for signatures
  ([`03556f0`](https://github.com/apduncan/cvanmf/commit/03556f06060b5ac4b4461c8534602f165645d373))

Gave Decomposition a `plot_weight_distribution` method, which produces a column plot of the feature
  weights in the signature.


## v0.4.1 (2024-10-22)

### Bug Fixes

- **denovo.signature_similarity**: Fix rank 1 only behaviour
  ([`16278f5`](https://github.com/apduncan/cvanmf/commit/16278f527607f9a877c51e812c70c56b1f62dc03))

changed `signature_similairty` to return an empty series when passed only rank 1 decompositions in
  line with `cophenetic_correlation` and `dispersion`. needed fix for a step in Nextflow pipeline
  when using k=1.

### Documentation

- *****: Fix some formatting
  ([`c5813f6`](https://github.com/apduncan/cvanmf/commit/c5813f6b987e768e8be393bac1ac6c9f97bfdb08))

- *****: Update installation instructions
  ([`f05d1f9`](https://github.com/apduncan/cvanmf/commit/f05d1f9022cfcb7da7a36e03cb584073d011a0cc))


## v0.4.0 (2024-09-30)

### Features

- **denovo**: Add post-hoc K-W tests ([#13](https://github.com/apduncan/cvanmf/pull/13),
  [`47d2eb5`](https://github.com/apduncan/cvanmf/commit/47d2eb5edebeb7c31aa686c3bcd7af95eefcdd4d))

Added post-hoc testing using Dunn's test to the `univariate_tests` method. This information is not
  yet displayed in the default plots.


## v0.3.2 (2024-09-27)

### Bug Fixes

- **denovo**: Indicate effect direction ([#12](https://github.com/apduncan/cvanmf/pull/12),
  [`e858fb5`](https://github.com/apduncan/cvanmf/commit/e858fb5f3d88bec51890524a08c73c7198dda1c4))

Added columns univariate_test output indicating which factor level has the maximum mean and median.
  For two group tests can use this as an indication of effect direction.


## v0.3.1 (2024-09-10)

### Bug Fixes

- **denovo**: Sort values by rank in stability rank select
  ([`391a2fa`](https://github.com/apduncan/cvanmf/commit/391a2fa8adf19894c2012bb4281c97be126f14b1))

If stability critera (coph, disp, sigsim) were not in rank order in input automated suggestions of
  rank would be incorrect. Internally sorts by rank now so can be provided in any order.

- **denovo**: Use online kneed
  ([`beaefae`](https://github.com/apduncan/cvanmf/commit/beaefae1090765dc02ba198a0dce24a6a8f32808))

Ocassionally too low a rank was being selected from bicrossvalidation curves, where a clear elbow
  point was visible later. Changed kneed to use online mode by default, which scan the entire curve
  rather than terminating at the first elbow.


## v0.3.0 (2024-09-05)

### Documentation

- Add summaries
  ([`4b48520`](https://github.com/apduncan/cvanmf/commit/4b48520fe37239f86a3347acc3594d35bf1e401d))

summaries for some module and overall package added.

### Features

- **stability**: Add methods to show stability of signatures
  ([#10](https://github.com/apduncan/cvanmf/pull/10),
  [`870cd84`](https://github.com/apduncan/cvanmf/commit/870cd842f15e8b5a8288226e091ff9df7cd827ab))

* feat(stability): allow use of signature similarity as rank selection method.


## v0.2.0 (2024-08-30)

### Bug Fixes

- *****: Tidy loggers
  ([`d918057`](https://github.com/apduncan/cvanmf/commit/d9180573d8bc4ce80e06d150a18db17fbd8b343f))

use module loggers rather than the global logging functions

- **data**: Include swimmer data
  ([`9ef625b`](https://github.com/apduncan/cvanmf/commit/9ef625bd1a6131ea29599d6d6a16da27acf7fd5f))

added swimmer data to repo.

- **denovo**: Allow custom colours for group in relative weight plot
  ([`3cf28ee`](https://github.com/apduncan/cvanmf/commit/3cf28eed3d08c7880bdcf327105a24fc5345d959))

`plot_relative_weight` now accepts parameter `group_colors` which is a series mapping metadata value
  to a colour for the ribbon element of the plot. can be blank for default Marsilea colours.

- **denovo**: Change to star suggested rank/alpha
  ([`6a23b2d`](https://github.com/apduncan/cvanmf/commit/6a23b2d9e203ffa21dbb7308d260e74a1b00897e))

`plot_rank_selection` and `plot_regu_selection` use a star at the top of the plot to indicate
  suggested rank rather than a dashed vertical line.

- **denovo**: Improve handling rank 1 by stability
  ([`ac80647`](https://github.com/apduncan/cvanmf/commit/ac806478158fa8628a2260d0897923d7a92dace4))

removes decompositions of rank 1 from calculation of stability based rank selection methods, as not
  meaningful.

- **denovo**: Load from tar.gz properly
  ([`8fd938e`](https://github.com/apduncan/cvanmf/commit/8fd938e03dfce95bc0f5f4241d2fac604f47dc2f))

- **denovo**: Remove rank 1 from stability rank suggestion.
  ([`1829a17`](https://github.com/apduncan/cvanmf/commit/1829a174e001b4e01dd86f98920a3e3bf267dfa0))

- **denovo**: Use consistent colours for bicv measures
  ([`bf8f91b`](https://github.com/apduncan/cvanmf/commit/bf8f91ba26acc1b00dfd2a8525120d348ff6cb2b))

`plot_rank_selection` and `plot_regu_selection` now uses the same colour for each property whenever
  displayed, doesn't always start at the same colour.

### Documentation

- *****: Micromamba troubleshooting, docstring
  ([`6bc8093`](https://github.com/apduncan/cvanmf/commit/6bc80936ae586386336ccdafb8ba94c144129bcf))

Fix method to make conda available to nextflow when using micromamba. Update some docstring
  formating to prevent newlines. Add semantic versioning to dev dependencies.

### Features

- **data**: Data subpackage added
  ([`ec89c6b`](https://github.com/apduncan/cvanmf/commit/ec89c6b680b7ef5f3f29abbd9c8d7e0d126e3939))

Add data subpackage, along with ALL/AML dataset and Swimmer dataset. Move synthetic data generation
  method to data subpackage. Add ExamleData class to hold data and metadata for provided/synthetic
  datasets.

- **data**: Nsclc example data and notebook added
  ([`010a8e5`](https://github.com/apduncan/cvanmf/commit/010a8e59b15687c11b78c33fe3668cd5391305a3))

add case study on NSCLC atlas to documentation, and included data in the data module.

- **denovo**: Change plotting backend
  ([`b095e84`](https://github.com/apduncan/cvanmf/commit/b095e840dcd082c78311ca71ae568b251bf563bd))

change relative weight plot to use marsilea remove patchwork lib update plotnine to 0.13.6

- **denovo**: Plot_metadata now can include modelfit
  ([`36e1f7d`](https://github.com/apduncan/cvanmf/commit/36e1f7dc945f25193ee96bfb019686886fe4f092))

add model fit to the standard metadata plots. this will eventually replace plot_modelfit, but for
  now both exists.

- **denovo**: Stability rank selection methods
  ([`c780607`](https://github.com/apduncan/cvanmf/commit/c7806077d14840264b8eb074b40ab087e7ccc95e))

Add cophenetic correlation and dispersion coefficients for rank selection and associated plots.

- **denovo**: Stability rank suggestion
  ([`5447016`](https://github.com/apduncan/cvanmf/commit/5447016086f079929ada01eeae3013bcc6f84d13))

add function to suggest ranks based on peaks for stability based measures (dispersion, cophenetic
  correlation)


## v0.1.0 (2024-07-18)
