import pandas as pd
import geopandas as gpd

def natural_gas_prices(prices_file, shapefile_path):
    prices = pd.read_excel(prices_file, skiprows=11, sheet_name='Sheet 1')
    prices = prices.rename(columns={'GEO (Labels)': 'nodes', 'Unnamed: 1': 'price_import'})
    prices = prices[['nodes', 'price_import']]
    prices['price_import'] = prices['price_import'] * 1000
    print(prices)
    nuts2 = gpd.read_file(shapefile_path).to_crs(epsg=3035)

    country_mapping = {
        'Belgium': 'BE',
        'Bulgaria': 'BG',
        'Czechia': 'CZ',
        'Denmark': 'DK',
        'Germany': 'DE',
        'Estonia': 'EE',
        'Ireland': 'IE',
        'Greece': 'GR',
        'Spain': 'ES',
        'France': 'FR',
        'Croatia': 'CR',
        'Italy': 'IT',
        'Latvia': 'LV',
        'Lithuania': 'LT',
        'Luxembourg': 'LU',
        'Hungary': 'HU',
        'Netherlands': 'NL',
        'Austria': 'AT',
        'Poland': 'PL',
        'Portugal': 'PT',
        'Romania': 'RO',
        'Slovenia': 'SI',
        'Slovakia': 'SK',
        'Finland': 'FI',
        'Sweden': 'SE'
    }

    prices['CNTR_CODE'] = prices['nodes'].map(country_mapping)

    nuts2 = nuts2.merge(prices, on='CNTR_CODE')

    nuts2 = nuts2[['NUTS_ID', 'CNTR_CODE', 'LEVL_CODE', 'price_import']]

    nuts2 = nuts2.loc[nuts2['LEVL_CODE'] == 2]
    price_import = nuts2[['NUTS_ID', 'price_import']]
    price_import = price_import.rename(columns={'NUTS_ID': 'nodes'})
    print(price_import)
    price_import.to_csv('../data/hard_to_abate_biomass_040624/set_carriers/natural_gas/price_import_high.csv', index=False)

def new_start_year(file):
    df = pd.read_csv(file)
    df.loc[df['year'] == 2023, 'price_import_yearly_variation'] = 1.00000
    print(df)
    df.to_csv('../data/hard_to_abate_biomass_040624/set_carriers/natural_gas/price_import_yearly_variation.csv', index=False)


shapefile_path = "../notebooks/nuts_data/NUTS_RG_20M_2021_4326.shp"
prices_file = "nrg_pc_203_c_page_spreadsheet.xlsx"
#natural_gas_prices(prices_file, shapefile_path)

new_start_year("../data/hard_to_abate/set_carriers/natural_gas/price_import_yearly_variation.csv")