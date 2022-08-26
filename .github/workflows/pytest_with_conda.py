name: PyTest with Conda

on: [push]

env:
  CACHE_NUMBER: 0  # increase to reset cache manually

jobs:
  build:
    
# setup the matrix for testing
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            label: linux-64
            prefix: /usr/share/miniconda3/envs/zen-garden

          - os: macos-latest
            label: osx-64
            prefix: /Users/runner/miniconda3/envs/zen-garden

# the testing
    name: ${{ matrix.label }}
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3 # checkout the repo

      # setup the conda env
      - name: Setup Mambaforge
        uses: conda-incubator/setup-miniconda@v2
        with:
            miniforge-variant: Mambaforge
            miniforge-version: latest
            activate-environment: zen-garden
            use-mamba: true

      # date for the cache
      - name: Set cache date
        run: echo "DATE=$(date +'%Y%m%d')" >> $GITHUB_ENV

      # reset cache if env file changed
      - uses: actions/cache@v2
        with:
          path: ${{ matrix.prefix }}
          key: ${{ matrix.label }}-conda-${{ hashFiles('zen_garden_env.yml') }}-${{ env.DATE }}-${{ env.CACHE_NUMBER }}
        id: cache

      # This step is cached in case the env file was not changed
      - name: Update environment
        run: mamba env update -f zen_garden_env.yml
        if: steps.cache.outputs.cache-hit != 'true'

      - name: Run tests
        shell: bash -l {0}
        run: pytest ./tests
