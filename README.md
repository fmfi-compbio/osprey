## Instalation

* Install trans-decoder (`https://github.com/fmfi-compbio/transducer_decoder`)
* Install mkl (`https://software.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup.html`)
* Set `export MKLROOT=<path_to_mkl>`
* `pip install .`

## Running

`osprey_basecaller.py --directory <directory_with_reads> --output <output_fasta_path> --threads <number_of_threads>`
