import pandas as pd

def fit_demands(demand_file):
    df = pd.read_csv(demand_file)
    df['2025'] = 1.0
    df.to_csv(f'../data/hard_to_abate/set_carriers/{industry}/{variation}_2.csv', index=False)
    print(df)



if __name__ == "__main__":

    industries = ['ammonia', 'methanol', 'oil_products', 'cement',  'steel']
    variations = ['demand_yearly_variation', 'demand_yearly_variation_high', 'demand_yearly_variation_low']

    for industry in industries:
        for variation in variations:
            demand_file = f'../data/hard_to_abate/set_carriers/{industry}/{variation}.csv'
            fit_demands(demand_file)

