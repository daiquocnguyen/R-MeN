<p align="center">
	<img src="https://github.com/daiquocnguyen/R-MeN/blob/master/rmen_logo.png">
</p>

# A Relational Memory-based Embedding Model for Triple Classification and Search Personalization<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FR-MeN%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/R-MeN"><a href="https://github.com/daiquocnguyen/R-MeN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/R-MeN"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/R-MeN">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/R-MeN">
<a href="https://github.com/daiquocnguyen/R-MeN/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/R-MeN"></a>
<a href="https://github.com/daiquocnguyen/R-MeN/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/R-MeN"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/R-MeN">

This program provides the implementation of the relational memory-based model R-MeN as described in [the paper]() where we integrate self-attetion mechanism-based memory interactions with a CNN encoder to effectively capture potential dependencies among relations and entities.
        
<p align="center">
	<img src="https://github.com/daiquocnguyen/R-Men/blob/master/rmen.png" width="350">
</p>

## Usage

### Requirements
- Python 3.x
- Tensorflow 1.14
- dm-sonnet 1.34
- scipy 1.3

### Running commands:

	$ python train_R_MeN_TripleCls_CNN.py --name WN11 --embedding_dim 50 --num_epochs 30 --batch_size 16 --head_size 1024 --memory_slots 1 --num_heads 3 --attention_mlp_layers 4 --num_filters 1024 --learning_rate 0.0001 --model_name wn11_d50_bs16_f1024_mlp4_hs1024_1_3_2
		
	$ python train_R_MeN_TripleCls_CNN.py --name FB13 --embedding_dim 50 --num_epochs 30 --batch_size 256 --head_size 1024 --memory_slots 1 --num_heads 2 --attention_mlp_layers 4 --num_filters 1024 --learning_rate 0.000005 --model_name fb13_d50_bs256_f1024_mlp4_hs1024_1_2_5

## Cite 

Please cite the paper whenever R-Men is used to produce published results or incorporated into other software:

	@InProceedings{Nguyen2019RMeN,
          author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
          title={{A Relational Memory-based Embedding Model for Triple Classification and Search Personalization}},
          booktitle={arxiv},
          year={2019}
          }

## License

As a free open-source implementation, R-Men is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

R-Men is licensed under the Apache License 2.0.
