## Introduction

Similarity-aware-NMT provides a general NMT framework to incorporate multiple dimensional translation memory based features as prior knowledge into Neural Machine Translation. Meanwhile, Similarity-aware-NMT preprocesses the testing set to get high-potential testing set, like string high-potential,structural high-potential and temporal high-potential Please refer to the following paper for details:

Tianfu Zhang, Heyan Huang, Chong Feng, Xiaochi Wei. 2020. [Similarity-aware neural machine translation: reducing human
translator efforts by leveraging high-potential sentences with translation memory](https://link.springer.com/content/pdf/10.1007/s00521-020-04939-y.pdf). Neural Computing and Applications, pp. 1–13.2020.

## Installation

Similarity-aware-NMT is built on top of [THUMT](http://github.com/thumt/THUMT). It requires THEANO 0.8.2 or above version (0.8.2 is recommended)

`` pip install theano==0.8.2 ``


## License

The source code is dual licensed. Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes. 
